# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import os
import sys
import pytorch_utils as pt_utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes,PointnetSAModuleVotes_seed,PointnetSAModuleVotes_seed2,PointnetSAModuleVotes_seed2_CMLP
import pointnet2_utils
from transformer import TransformerDecoderLayer
from modules import PositionEmbeddingLearned

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.nsample = 24
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=self.nsample,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        self.aggregation_vote2seed = PointnetSAModuleVotes_seed2( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=self.nsample,
                mlp=[self.seed_feat_dim, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )    
                  
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(512,256,1)
        self.conv11 = torch.nn.Conv1d(256,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn11 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.conv_pos1 = torch.nn.Conv1d(3,256,1)
        self.conv_pos2 = torch.nn.Conv1d(256,256,1)
        self.bn_pos1 = torch.nn.BatchNorm1d(256)

        self.num_decoder_layers = 8
        self.nhead = 4
        self.dim_feedforward = 1024
        self.dropout = 0.1
        self.activation = "relu"
        self.decoder = nn.ModuleList()
        self.decoder_self_posembeds = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 256))

        self.decoder_cross_posembeds = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 256))

        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    256, self.nhead, self.dim_feedforward, self.dropout, self.activation ,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))        
        self.decoder_query_proj = nn.Conv1d(256, 256, kernel_size=1)
        self.decoder_key_proj = nn.Conv1d(256, 256, kernel_size=1)
        self.decoder_value_proj = nn.Conv1d(256, 256, kernel_size=1)
    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        ############################
        features_seed1024 = end_points['seed_features_2048']
        features_vote1024 = end_points['vote_features_2048']
        seed_xyz = end_points['seed_xyz']
        seed_feature = end_points['seed_features']
        vote_weight_features_2048 = end_points['vote_weight_features_2048']
        seed_weight_features_2048 = end_points['seed_weight_features_2048']
        vote_xyz_2048 = end_points['vote_xyz_2048']
        seed_xyz_2048 = end_points['seed_xyz_2048']
        xyz = vote_xyz_2048
        features = features_vote1024
        seed_xyz = seed_xyz_2048
        seed_feature = features_seed1024
        batch_size = seed_xyz.shape[0]

        ############################
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds,queryseedidx = self.aggregation_vote2seed(xyz, features)

            sample_inds = fps_inds  
            grouped_feature_seed = pointnet2_utils.grouping_operation(seed_feature, queryseedidx)
            grouped_feature_xyz = pointnet2_utils.grouping_operation(seed_xyz.transpose(1, 2).contiguous(), queryseedidx)
            #point: [B,3,256,16]
            #feature: [B,256,256,16]
            grouped_feature_seed_tr = grouped_feature_seed.transpose(1,2) # [B,N,C,n]
            #grouped_feature_seed_tr =   grouped_feature_seed_tr.reshape(batch_size*256,256,16)
            grouped_feature_seed_tr = rearrange(grouped_feature_seed_tr, 'b h w c -> (b h) w c')
            # (b*256, c, n)
            grouped_feature_xyz_tr = grouped_feature_xyz.transpose(1,2)# [B,N,3,n]
            grouped_feature_xyz_tr = rearrange(grouped_feature_xyz_tr, 'b h w c -> (b h) w c')
            
            #grouped_feature_xyz_tr =   grouped_feature_xyz_tr.reshape(batch_size*256,3,16)
            grouped_feature_xyz_tr = grouped_feature_xyz_tr.transpose(1,2)
            # (b*256, n, 3)

            features_tr = features.transpose(1,2)
            features_tr = rearrange(features_tr, 'b h w  -> (b h) w ').unsqueeze(-1)            

            token_0 =  torch.zeros_like(features_tr).cuda()  
            query = torch.cat((token_0, grouped_feature_seed_tr), dim=-1)  # ([2048, 256, 17])

            #query = torch.cat((features_tr, grouped_feature_seed_tr), dim=-1)  # ([2048, 256, 17])
            key =query
            value = query

            query = self.decoder_query_proj(query) 
            key = self.decoder_key_proj(key)
            value = self.decoder_value_proj(value)

            xyz_tr =  rearrange(xyz, 'b h w  -> (b h) w ').unsqueeze(-1).transpose(1,2)
            xyz_tr_0 = torch.zeros_like(xyz_tr).cuda()  

            xyz_tr_24 = xyz_tr.repeat(1,self.nsample,1)

            xyz_detlt = grouped_feature_xyz_tr -xyz_tr_24
            

            zreo = xyz_tr_0
         
            xyz_detlt = torch.cat((zreo, xyz_detlt), dim=1)
            
            # query = query+xyz_detlt
            # key = query

            #print("xyz_detlt",xyz_detlt.shape)
            #grouped_feature_xyz_tr_g_l = grouped_feature_xyz_tr +xyz_detlt

            query_pos = torch.cat((xyz_tr, grouped_feature_xyz_tr), dim=1)    
            
            key_pos = query_pos 
            query_pos_scenes = query_pos
            query_pos_object = xyz_detlt
            for i in range(self.num_decoder_layers):   

                query = self.decoder[i](query, key,value, query_pos=query_pos_scenes,
                key_pos=query_pos_object)
            #print("query",query.shape)
            feat = query[:,:,0]
            #print("feat",feat.shape)
            feat =  rearrange(feat, '(b h) w  -> b h w ', b=batch_size, w=256)
            #features = feat.transpose(1,2)+features
            features = torch.cat((features,feat),1)
            #print("feat",feat.shape)



        ############################
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(seed_xyz, self.num_proposal)
            xyz, features, fps_inds,queryseedidx = self.aggregation_vote2seed(xyz, features, sample_inds)
            #seed_xyz = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), queryseedidx).transpose(1, 2).contiguous() 
            sample_inds = fps_inds
            
            grouped_feature_seed = pointnet2_utils.grouping_operation(seed_feature, queryseedidx)
            grouped_feature_xyz = pointnet2_utils.grouping_operation(seed_xyz.transpose(1, 2).contiguous(), queryseedidx)
            #point: [B,3,256,16]
            #feature: [B,256,256,16]
            grouped_feature_seed_tr = grouped_feature_seed.transpose(1,2) # [B,N,C,n]
            #grouped_feature_seed_tr =   grouped_feature_seed_tr.reshape(batch_size*256,256,16)
            grouped_feature_seed_tr = rearrange(grouped_feature_seed_tr, 'b h w c -> (b h) w c')
            # (b*256, c, n)
            grouped_feature_xyz_tr = grouped_feature_xyz.transpose(1,2)# [B,N,3,n]
            grouped_feature_xyz_tr = rearrange(grouped_feature_xyz_tr, 'b h w c -> (b h) w c')
            
            #grouped_feature_xyz_tr =   grouped_feature_xyz_tr.reshape(batch_size*256,3,16)
            grouped_feature_xyz_tr = grouped_feature_xyz_tr.transpose(1,2)
            # (b*256, n, 3)

            features_tr = features.transpose(1,2)
            features_tr = rearrange(features_tr, 'b h w  -> (b h) w ').unsqueeze(-1)            

            token_0 =  torch.zeros_like(features_tr).cuda()  
            query = torch.cat((token_0, grouped_feature_seed_tr), dim=-1)  # ([2048, 256, 17])

            #query = torch.cat((features_tr, grouped_feature_seed_tr), dim=-1)  # ([2048, 256, 17])
            key =query
            value = query

            query = self.decoder_query_proj(query) 
            key = self.decoder_key_proj(key)
            value = self.decoder_value_proj(value)

            xyz_tr =  rearrange(xyz, 'b h w  -> (b h) w ').unsqueeze(-1).transpose(1,2)
            xyz_tr_0 = torch.zeros_like(xyz_tr).cuda()  

            xyz_tr_24 = xyz_tr.repeat(1,24,1)

            xyz_detlt = grouped_feature_xyz_tr -xyz_tr_24
            

            zreo = xyz_tr_0
         
            xyz_detlt = torch.cat((zreo, xyz_detlt), dim=1)
            
            # query = query+xyz_detlt
            # key = query

            #print("xyz_detlt",xyz_detlt.shape)
            #grouped_feature_xyz_tr_g_l = grouped_feature_xyz_tr +xyz_detlt

            query_pos = torch.cat((xyz_tr, grouped_feature_xyz_tr), dim=1)    
            
            key_pos = query_pos 
            query_pos_scenes = query_pos
            query_pos_object = xyz_detlt
            for i in range(self.num_decoder_layers):   

                query = self.decoder[i](query, key,value, query_pos=query_pos_scenes,
                key_pos=query_pos_object)
            #print("query",query.shape)
            feat = query[:,:,0]
            #print("feat",feat.shape)
            feat =  rearrange(feat, '(b h) w  -> b h w ', b=batch_size, w=256)
            #features = feat.transpose(1,2)+features
            features = torch.cat((features,feat),1)
            #print("feat",feat.shape)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        
                
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn11(self.conv11(net)))
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)
        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
