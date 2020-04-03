import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hpcnet_utils
from pointnet2 import pointnet2_utils
from pointnet2 import pytorch_utils as pt_utils
from typing import List


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'
        self.new_xyz = None

    def forward(self, xyz: torch.Tensor, points: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        # """
        # :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        # :param features: (B, N, C) tensor of the descriptors of the the features
        # :param new_xyz:
        # :return:
        #     new_xyz: (B, npoint, 3) tensor of the new features' xyz
        #     new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        # """
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # xyz = xyz.permute(0, 2, 1).contiguous()
        # if points is not None:
        #     points = points.permute(0, 2, 1)
        if points is not None:
            features = points[:,3:,:].contiguous()
        else:
            features = None

        new_features_list = []

        # xyz_flipped = xyz.transpose(1, 2).contiguous()
        xyz_flipped = xyz.contiguous()
        xyz = xyz.transpose(1,2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        new_xyz = new_xyz.contiguous()
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz.contiguous(), features)  # (B, C, npoint, nsample)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        new_xyz = new_xyz.permute(0,2,1)
        return new_xyz, torch.cat(new_features_list, dim=1)

class HPC_SAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, npoint: int, radii: List[float], nsamples: List[int], in_channel: int, mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool', instance_norm=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                hpcnet_utils.HPC_Group(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            # mlp_spec[0] += 42 + in_channel
            mlp_spec.insert(0, 42+in_channel)
            # if use_xyz:
            #     mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
        self.pool_method = pool_method


if __name__ == "__main__":
    pass
