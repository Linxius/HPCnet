import torch
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

theads_num = 32

class getGtFeature(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, keyclouds: torch.Tensor, \
                grouped_xyz: torch.Tensor, radius: float) -> torch.Tensor:
        """
        points: B C N
        keyclouds: B C N
        grouped_xyz: nsamples B C N
        """
        voxel_num_1dim = 30
        voxel_len = radius / voxel_num_1dim

        root = "./HausdorffTest/shapes/" + str(radius)
        vKeyshapes, vShapedicts = LoadGivenShapes(root)
        gt_feature_len = len(vShapedicts)

        batch_size = keyclouds.size()[0]
        point_dim = keyclouds.size()[1]
        point_num = keyclouds.size()[2]
        points = points.share_memory_()
        keyclouds = keyclouds.share_memory_()
        feature = torch.zeros(batch_size, len(vKeyshapes), point_num,requires_grad=False).share_memory_()
        grouped_xyz = grouped_xyz.permute(1, 2, 3, 0).share_memory_()

        r2 = radius ** 2
        for k in range(batch_size):
            clouds = points[k,:,:].transpose(0,1)
            keypoints = keyclouds[k,:,:].transpose(0,1)

            step = int(point_num / theads_num)
            for thread_index in range(theads_num):
                mp.spawn(point_feature_worker, nprocs=4, args=(points, keyclouds, grouped_xyz, feature, step,\
                                                     thread_index, batch_size, point_num, theads_num, k,\
                                                     radius, voxel_len, voxel_num_1dim, gt_feature_len,\
                                                     vKeyshapes, vShapedicts))

        return feature

    @staticmethod
    def backward(ctx, a = None):
        return None, None, None, None

def point_feature_worker(proc, points, keyclouds, grouped_xyz, feature, step,\
                         thread_index, batch_size, point_num, theads_num, k,\
                         radius, voxel_len, voxel_num_1dim, gt_feature_len,\
                         vKeyshapes, vShapedicts):
    for i in range(step):
        points_neighbor = grouped_xyz[k,:,i,:].transpose(0,1)

        for j in range( gt_feature_len ):
            fToTempDis = -1.0*sys.float_info.max
            for it in range(len(points_neighbor)):
                ith = math.floor(abs(points_neighbor[it][0] + radius) / voxel_len)
                jth = math.floor(abs(points_neighbor[it][1] + radius) / voxel_len)
                kth = math.floor(abs(points_neighbor[it][2] + radius) / voxel_len)
                # TODO 下标出错临时解决
                ith = ith if ith < 30 else 29
                jth = jth if jth < 30 else 29
                kth = ith if kth < 30 else 29
                iOneIdx = ith + jth * voxel_num_1dim + kth * voxel_num_1dim * voxel_num_1dim
                fOneDis = vShapedicts[j][iOneIdx]
                if fToTempDis < fOneDis:
                    fToTempDis = fOneDis

            fToSourceDis = 0
            for it in range(len(vKeyshapes[j])):
                for iit in range(points_neighbor.size()[0]):
                    oneneardis = ((vKeyshapes[j][it][0]-points_neighbor[iit][0])**2 + \
                                    (vKeyshapes[j][it][0]-points_neighbor[iit][0])**2 + \
                                    (vKeyshapes[j][it][0]-points_neighbor[iit][0])**2)
                    if fToSourceDis < oneneardis:
                        fToSourceDis = oneneardis
            fToSourceDis = math.sqrt(fToSourceDis)

            fGenHdis = fToTempDis if fToTempDis > fToSourceDis else fToSourceDis
            fGenHdis = fGenHdis/radius
            if fGenHdis > 1.0:
                fGenHdis = 1.0
            feature[k, j, i] = fGenHdis

get_gt_feature = getGtFeature.apply
