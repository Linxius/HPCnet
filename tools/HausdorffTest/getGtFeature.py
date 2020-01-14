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
        root = "./HausdorffTest/shapes/" + str(radius)
        vKeyshapes, vShapedicts = LoadGivenShapes(root)
        gt_feature_len = len(vShapedicts)

        voxel_num_1dim = 30
        voxel_len = 2*radius / voxel_num_1dim
        voxel_num_1dim = int(2*radius/voxel_len + 1)

        batch_size = keyclouds.size()[0]
        point_dim = keyclouds.size()[1]
        point_num = keyclouds.size()[2]

        points = points.share_memory_()
        keyclouds = keyclouds.share_memory_()
        feature = torch.zeros(batch_size, len(vKeyshapes), point_num,requires_grad=False).share_memory_()
        grouped_xyz = grouped_xyz.permute(1, 2, 3, 0).share_memory_()
        vKeyshapes = torch.Tensor(vKeyshapes).share_memory_()
        vShapedicts = torch.Tensor(vShapedicts).share_memory_()

        r2 = radius ** 2
        for k in range(batch_size):
            for i in range(point_num):
                points_neighbor = grouped_xyz[k,:,i,:].transpose(0,1)

                for j in range( gt_feature_len ):
                    fToTempDis = 0
                    for it in range(len(points_neighbor)):
                        ith = math.floor(abs(points_neighbor[it][0] + radius) / voxel_len)
                        jth = math.floor(abs(points_neighbor[it][1] + radius) / voxel_len)
                        kth = math.floor(abs(points_neighbor[it][2] + radius) / voxel_len)
                        iOneIdx = int(ith + jth * voxel_num_1dim + kth * voxel_num_1dim * voxel_num_1dim)
                        fOneDis = vShapedicts[j][iOneIdx]
                        if fToTempDis < fOneDis:
                            fToTempDis = fOneDis

                    fToSourceDis = 0
                    for it in range(vKeyshapes[j].shape[0]):
                        minPointPairdis = 99.9
                        for iit in range(points_neighbor.size()[0]):
                            oneneardis = ((vKeyshapes[j][it][0]-points_neighbor[iit][0])**2 + \
                                            (vKeyshapes[j][it][1]-points_neighbor[iit][1])**2 + \
                                            (vKeyshapes[j][it][2]-points_neighbor[iit][2])**2)
                            if minPointPairdis > oneneardis:
                                minPointPairdis = oneneardis
                        if fToSourceDis < minPointPairdis:
                            fToSourceDis = minPointPairdis
                    fToSourceDis = math.sqrt(fToSourceDis)
                    fGenHdis = fToTempDis if fToTempDis > fToSourceDis else fToSourceDis
                    fGenHdis = 1.0 if fGenHdis > radius else fGenHdis / radius
                    feature[k, j, i] = 1 - fGenHdis
                # print(feature[k,:,i])

        return feature

    @staticmethod
    def backward(ctx, a = None):
        return None, None, None, None

get_gt_feature = getGtFeature.apply
