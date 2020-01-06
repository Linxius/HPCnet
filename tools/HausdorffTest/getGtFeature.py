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

def Outputrestxt(filename,res):
    with open(filename,"a+") as f:
        for row in res:
            line = str(row).replace('[','').replace(']','')
            #line = line.replace("'",'').replace(',','')+'\n'
            line = line.replace("'",'').replace(',','')+' '
            f.writelines(line)

gt_feature_len = 42
theads_num = 8
def getGtFeature(points, radius):
    # radius = 0.1
    #HausdorffOp is Hausdorff Operation (class object)
    HausdorffOp = Haus.Hausdorff(radius,radius/15.0)
    root = "./HausdorffTest/shapescales/" + str(radius)
    # print(root)
    # import pdb; pdb.set_trace()
    # vKeyshapes, vShapedicts = ReadShapes.LoadGivenShapes(root)
    vKeyshapes, vShapedicts = LoadGivenShapes(root)
    gt_feature_len = len(vShapedicts)

    batch_size = points.size()[0]
    point_dim = points.size()[1]
    point_num = points.size()[2]
    feature = torch.zeros(batch_size, len(vKeyshapes), point_num,requires_grad=False)

    pool = multiprocessing.Pool(theads_num)

    for batch_index in range(theads_num):
        pool.apply_async(thread_compute, (points, batch_index, feature, batch_size, theads_num))

    pool.close()
    pool.join()

    # return torch.cat([points, feature], 1 )
    return feature

def thread_compute(points, batch_index, feature, batch_size, theads_num):
    for k in range(int(batch_size/theads_num)):
        k = k + batch_index
        clouds = points[k,:,:].transpose(0,1).numpy().tolist()

        #build a kd-tree
        srctree = spatial.KDTree(data = clouds)
        keyclouds = clouds
        #compute each key points neighboring shape

        for i in range(len(keyclouds)):
            neighidxs = srctree.query_ball_point(keyclouds[i],radius)

            points_neighbor = HausdorffOp.RelativeCor(clouds, neighidxs, keyclouds[i])
            #build a kdtree for neighboring point clouds
            #it will be used in computing Hausdorff distance from template to source
            neighbortree = spatial.KDTree(points_neighbor)

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
