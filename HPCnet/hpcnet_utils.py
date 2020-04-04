import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import pointnet2_cuda as pointnet2

from HPCnet.getGtFeature import get_gt_feature
from pointnet2.pointnet2_utils import ball_query, grouping_operation


class HPC_Group(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        gtfeatures = get_gt_feature(xyz, new_xyz, \
                                    grouped_xyz.permute(0,2,3,1).contiguous(),\
                                    self.radius, self.nsample).transpose(1,2) # 8 42 4096
        gtfeatures = gtfeatures.unsqueeze(-1).expand(-1,-1,-1,self.nsample)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, \
                                          gtfeatures, grouped_features], \
                                         dim=1)  # (B, C + 3, npoint, nsample)
            else:
                # new_features = grouped_features
                new_features = torch.cat([gtfeatures, \
                                          grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = torch.cat([grouped_xyz, \
                                      gtfeatures], dim=1)  # (B, C + 3, npoint, nsample)
        return new_features
