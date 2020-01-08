#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:43:11 2019

@author: ludy
"""

import numpy as np
import torch
from scipy import spatial
from threading import Thread
from HausdorffTest.ReadShapes import read_keyfile
from HausdorffTest.ReadShapes import LoadGivenShapes
import time
import os
import HausdorffTest.Hausdorff as Haus
import multiprocessing
import sys
sys.path.append('..')
# from pointnet2.pointnet2_utils import BallQuery, GroupingOperation
# ball_query = pointnet2.pointnet2_utils.BallQuery.apply
# grouping_operation = GroupingOperation.apply
from pointnet2.pointnet2_utils import grouping_operation, ball_query

gt_feature_len = 42
# theads_num = 8
theads_num = 8


"""
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
"""

def getGtFeature(points, keyclouds, grouped_xyz, nsample, radius):
    # radius = 0.1
    #HausdorffOp is Hausdorff Operation (class object)
    HausdorffOp = Haus.Hausdorff(radius,radius/15.0)
    root = "./HausdorffTest/shapes/" + str(radius)
    # print(root)
    # import pdb; pdb.set_trace()
    # vKeyshapes, vShapedicts = ReadShapes.LoadGivenShapes(root)
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

    pool = multiprocessing.Pool(theads_num)

    for batch_index in range(theads_num):
        pool.apply_async(thread_compute, (points, keyclouds, grouped_xyz, batch_index, radius, feature, \
        # pool.apply_async(thread_compute, (points, batch_index, radius, feature, \
                                          batch_size, point_num, theads_num, HausdorffOp, vKeyshapes, vShapedicts))

    pool.close()
    pool.join()

    # return torch.cat([points, feature], 1 )
    print(feature[0,:,10])
    print(feature.size())
    import pdb; pdb.set_trace()
    return feature

def thread_compute(points, keyclouds, grouped_xyz, batch_index, radius, feature, \
# def thread_compute(points, batch_index, radius, feature, \
                   batch_size, point_num, theads_num, HausdorffOp, vKeyshapes, vShapedicts):
    step = int(batch_size/theads_num)
    for k in range(step):
        k = k + batch_index * step
        # clouds = points[k,:,:].transpose(0,1).numpy().tolist()
        # keyclouds = keyclouds[k,:,:].transpose(0,1).numpy().tolist()

        #build a kd-tree
        # srctree = spatial.KDTree(data = clouds)
        # keyclouds = clouds
        #compute each key points neighboring shape

        # for i in range(len(keyclouds)):
        for i in range(point_num):
            # neighidxs = srctree.query_ball_point(keyclouds[i],radius)

            # points_neighbor = HausdorffOp.RelativeCor(clouds, neighidxs, keyclouds[i])
            points_neighbor = grouped_xyz[k, :, i, :].transpose(0,1).numpy().tolist()

            #build a kdtree for neighboring point clouds
            #it will be used in computing Hausdorff distance from template to source
            neighbortree = spatial.KDTree(points_neighbor)
            # print(type(neighbortree))
            # import pdb; pdb.set_trace()

            # vResCheck = []

            for j in range( gt_feature_len ):
                #compute Hausdorff distance from source to template
                fToTempDis = HausdorffOp.HausdorffDict(vCloud = points_neighbor, vDisDict = vShapedicts[j])

                #compute Hausdorff distance from template to source
                fToSourceDis = HausdorffOp.HausdorffDisMax(vKeyshapes[j], neighbortree)

                #compute the general Hausdorff distance, which is a two-side measurement
                fGenHdis = HausdorffOp.GeneralHausDis(fToTempDis, fToSourceDis)

                # vResCheck.append(fGenHdis)
                feature[k, j, i] = fGenHdis
                # print(fGenHdis)
                # import pdb; pdb.set_trace()
