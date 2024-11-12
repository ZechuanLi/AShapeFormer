# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from models.loss_helper import get_loss
import pointnet2_utils 
class GeneralSamplingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, features, sample_inds):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
        new_features = pointnet2_utils.gather_operation(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds
class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)
        
        # seg 
        self.seg1 = torch.nn.Conv1d(256, 128, 1)
        self.seg2 = torch.nn.Conv1d(128, 64, 1)
        self.seg3 = torch.nn.Conv1d(64, 1, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)
        self.gsample_module = GeneralSamplingModule()
    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']  # seed points
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['seed_features'] = features
       
              
        xyz, features = self.vgen(xyz, features) # vote_xyz vote_feature
        relu = nn.ReLU()
        #### semantic segmentation
        semantic_scores = self.seg3(relu(self.bn2(self.seg2(relu(self.bn1(self.seg1(end_points['seed_features'])))))))  # (N, nClass), float
        #semantic_scores = F.lonvidig_softmax(semantic_scores, dim=1)
        #semantic_scores = semantic_scores.squeeze(1)
        # semantic_preds = semantic_scores.max(1)[1]    # (N), long
        end_points['semantic_preds'] = semantic_scores
        #print("semantic_scores",semantic_scores.shape)
        score_pred =  torch.sigmoid(semantic_scores).squeeze(1)
        #print("score_pred",score_pred.shape)
        #score_pred_max, class_pred = score_pred.max(dim=1)
        sample_inds = torch.topk(score_pred, 1024)[1].int()
        scores_topk = torch.topk(score_pred, 1024)[0]      
        xyz_seed1024, features_seed1024, sample_inds = self.gsample_module(end_points['seed_xyz'], end_points['seed_features'], sample_inds)
        xyz_vote1024, features_vote1024, sample_voteinds = self.gsample_module(xyz, features, sample_inds)

        score_pred_obj = scores_topk
        score_pred_obj = score_pred_obj.unsqueeze(-1)
        score_pred_obj_256 = score_pred_obj.repeat(1,1,256)     
        score_pred_obj_256 = score_pred_obj_256.transpose(1,2) 
        #score_pred_obj = score_pred[:,1,:] # 预测为前景点的分数
        #score_picked, sample_idx = torch.topk(score_pred_obj, 100, dim=1) 
        score_pred_obj = score_pred.unsqueeze(-1)

        # features_norm = torch.norm(features, p=2, dim=1)
        # features = features.div(features_norm.unsqueeze(1))

        # features_norm_seed = torch.norm(features_seed1024, p=2, dim=1)
        # features_seed1024 = features_seed1024.div(features_norm_seed.unsqueeze(1))

        # features_norm_vote = torch.norm(features_vote1024, p=2, dim=1)
        # features_vote1024 = features_vote1024.div(features_norm_vote.unsqueeze(1))

        

        end_points['seed_features_2048']= features_seed1024
        end_points['seed_xyz_2048']= xyz_seed1024
        end_points['vote_xyz_2048'] = xyz_vote1024

        end_points['vote_features_2048'] = features_vote1024
        end_points['seed_weight_features_2048'] = features_seed1024*score_pred_obj_256
        end_points['vote_features_weight'] = score_pred_obj

        end_points['vote_weight_features_2048'] = features_vote1024*score_pred_obj_256
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)

        return end_points


if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from models.loss_helper_kps import get_loss

    # Define model
    model = VoteNet(10,12,10,np.random.random((10,3))).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key])

    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')



