import torch
from typing import Tuple
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import math
from HausdorffTest.ReadShapes import read_keyfile
from HausdorffTest.ReadShapes import LoadGivenShapes
import time
import os
import sys
import torch.multiprocessing as mp

import numpy as np
import HPCnet_cuda as HPCnet

class getGtFeature(Function):
    @staticmethod
    def forward(ctx, whole_points: torch.Tensor, keypoints: torch.Tensor, \
                neighbor_points: torch.Tensor, radius: float, neighbor_point_num: float)\
                -> torch.Tensor:
        """
        whole_points: B C N
        keypoints: B N C
        neighbor_points: B N nsample C
        output: feature: B M gt_num
        """
        root = "./HausdorffTest/shapes/" + str(radius)
        prior_points, dis_dicts = LoadGivenShapes(root)

        dis_dicts = torch.cuda.FloatTensor(dis_dicts)
        prior_points = torch.cuda.FloatTensor(prior_points)
        # gt_feature_len = len(dis_dicts)

        voxel_dim = 30
        voxel_len = 2*radius / voxel_dim
        voxel_dim = int(2*radius/voxel_len + 1)

        batch_size, keypoint_num, point_dim= keypoints.size()
        whole_point_num = whole_points.size()[0]

        # feature = torch.zeros(batch_size, len(prior_shapes), point_num, requires_grad=False)
        feature = torch.cuda.FloatTensor(batch_size, keypoint_num, len(prior_points)).zero_()
        # print(type(feature))
        # print(type(dis_dicts))

        # import pdb; pdb.settrace()
        HPCnet.get_hausdorff_dis_wrapper(whole_points, keypoints, neighbor_points, feature, radius,\
                                            batch_size, \
                                            whole_point_num, keypoint_num, neighbor_point_num, \
                                            prior_points, dis_dicts,\
                                            voxel_len)

        with open("./output/keypoints.txt", 'a') as keypoints_file:
            np.savetxt(keypoints_file, keypoints[0,200:256,:].cpu().detach().numpy())
        with open("./output/neighbor_points.txt", 'a') as neighbor_points_file:
            for i in range(200, 256):
                np.savetxt(neighbor_points_file, neighbor_points[0,i,:,:].cpu().detach().numpy())
        with open("./output/feature.txt", 'a') as feature_file:
            np.savetxt(feature_file, feature[0,200:256,:].cpu().detach().numpy())

        # print(feature[0,100:120,:])
        # import pdb; pdb.set_trace()
        return feature

    # @staticmethod
    # def backward(feature, a = None):
    #     return None, None, None, None, None

get_gt_feature = getGtFeature.apply
