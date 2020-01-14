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

import pointnet2_cuda as pointnet2

class getGtFeature(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, keyclouds: torch.Tensor, \
                grouped_xyz: torch.Tensor, radius: float, nsample: float) -> torch.Tensor:
        """
        points: B C N
        keyclouds: B C N
        grouped_xyz: B N nsample C
        """
        root = "./HausdorffTest/shapes/" + str(radius)
        vKeyshapes, vShapedicts = LoadGivenShapes(root)
        # prior_shapes, dis_dicts = LoadGivenShapes(root)
        gt_feature_len = len(vShapedicts)
        voxel_num_1dim = 30
        voxel_len = 2*radius / voxel_num_1dim
        voxel_num_1dim = int(2*radius/voxel_len + 1)

        batch_size = keyclouds.size()[0]
        point_dim = keyclouds.size()[1]
        point_num = keyclouds.size()[2]

        print(torch.Tensor(vKeyshapes).size())
        print(torch.Tensor(vShapedicts).size())
        import pdb; pdb.set_trace()
        # feature = torch.zeros(batch_size, len(vKeyshapes), point_num, requires_grad=False)
        feature = torch.cuda.FloatTensor(batch_size, point_num, len(vKeyshapes))

        pointnet2.get_hausdorff_dis_wrapper(points, keyclouds, grouped_xyz, feature, radius,\
                                            batch_size, point_dim, point_num, nsample, \
                                            torch.Tensor(vKeyshapes), torch.Tensor(vShapedicts), \
                                            voxel_len)#,\
                                            #gt_feature_len, voxel_num_1dim)

        return feature

    @staticmethod
    def backward(ctx, a = None):
        return None, None, None, None

get_gt_feature = getGtFeature.apply
