import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import pointnet2_cuda as pointnet2

class getNeighborsR(Function):
# class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, neighbors_num) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

get_neighbors_r = getNeighborsR.apply

class getMaxDis(Function):

    @staticmethod
    def forward(ctx, points: torch.Tensor, points_neighbor: torch.Tensor) -> float:
        return pointnet2.get_max_dis_wrapper(points, points_neighbor)

    @staticmethod
    def backward(ctx, a=None):
        return None, None


get_max_dis = getMaxDis.apply
