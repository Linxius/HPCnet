#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:43:11 2019

@author: ludy
"""

import numpy as np
import torch
from scipy import spatial
from HausdorffTest.ReadShapes import read_keyfile
from HausdorffTest.ReadShapes import LoadGivenShapes
import time
import os
import HausdorffTest.Hausdorff as Haus
# import sys
# sys.path.append('..')
# from pointnet2.pointnet2_utils import grouping_operation, ball_query

gt_feature_len = 42
theads_num = 8

def getGtFeature(points, keyclouds, grouped_xyz, nsample, radius):
    HausdorffOp = Haus.Hausdorff(radius,radius/15.0)
    root = "./HausdorffTest/shapes/" + str(radius)
    vKeyshapes, vShapedicts = LoadGivenShapes(root)
    gt_feature_len = len(vShapedicts)

    batch_size = keyclouds.size()[0]
    point_dim = keyclouds.size()[1]
    point_num = keyclouds.size()[2]
    feature = torch.zeros(batch_size, len(vKeyshapes), point_num,requires_grad=False)

    # print(keyclouds.size()) # B C N
    # print(points.size()) # B C N
    # print(grouped_xyz.size()) # B C N Nsample
    # import pdb; pdb.set_trace()

    for k in range(batch_size):
        clouds = points[k,:,:].transpose(0,1).numpy().tolist()
        srctree = spatial.KDTree(data = clouds)
        keypoints = keyclouds[k,:,:].transpose(0,1).numpy().tolist()
        for i in range(point_num):
            neighidxs = srctree.query_ball_point(keypoints[i],radius)
            points_neighbor = HausdorffOp.RelativeCor(clouds, neighidxs, keypoints[i])
            # points_neighbor = grouped_xyz[k, :, i, :].transpose(0,1).numpy().tolist()
            neighbortree = spatial.KDTree(points_neighbor)
            for j in range( gt_feature_len ):
                #compute Hausdorff distance from source to template
                fToTempDis = HausdorffOp.HausdorffDict(vCloud = points_neighbor, vDisDict = vShapedicts[j])

                #compute Hausdorff distance from template to source
                fToSourceDis = HausdorffOp.HausdorffDisMax(vKeyshapes[j], neighbortree)

                #compute the general Hausdorff distance, which is a two-side measurement
                fGenHdis = HausdorffOp.GeneralHausDis(fToTempDis, fToSourceDis)

                # vResCheck.append(fGenHdis)
                # feature[k, j, i] = 1 - fGenHdis / 2
                feature[k, j, i] = fGenHdis
                # print(fGenHdis)
                # import pdb; pdb.set_trace()
            # print(feature[k,:,i])

    # return torch.cat([points, feature], 1 )
    # print(feature)
    # import pdb; pdb.set_trace()
    return feature
