import torch
import math
from scipy import spatial
from threading import Thread
from HausdorffTest.ReadShapes import read_keyfile
from HausdorffTest.ReadShapes import LoadGivenShapes
import time
import os
import HausdorffTest.Hausdorff as Haus
import multiprocessing

gt_feature_len = 42
theads_num = 8

def getGtFeature(points, keyclouds, grouped_xyz, radius):
    """
    points: B C N
    keyclouds: B C N
    grouped_xyz: B C N nsamples
    """
    root = "./HausdorffTest/shapes/" + str(radius)
    vKeyshapes, vShapedicts = LoadGivenShapes(root)
    gt_feature_len = len(vShapedicts)
    voxel_num_1dim = 30
    voxel_len = 2*radius / voxel_num_1dim

    batch_size = keyclouds.size()[0]
    point_dim = keyclouds.size()[1]
    point_num = keyclouds.size()[2]
    feature = torch.zeros(batch_size, len(vKeyshapes), point_num,requires_grad=False)

    pool = multiprocessing.Pool(theads_num)

    for batch_index in range(theads_num):
        pool.apply_async(thread_compute, (points, keyclouds, grouped_xyz, feature, \
                                          batch_index, batch_size, point_num, theads_num, \
                                          radius, voxel_len, voxel_num_1dim, \
                                          vKeyshapes, vShapedicts))

    pool.close()
    pool.join()

    return feature


def thread_compute(points, keyclouds, grouped_xyz, feature, batch_index, batch_size, point_num,
                   threads_num, radius, voxel_len, voxel_num_1dim, vKeyshapes, vShapedicts):
    step = int(batch_size/theads_num)
    for k in range(step):
        k = k + batch_index * step
        for i in range(point_num):
            points_neighbor = grouped_xyz[k,:,i,:].transpose(0,1)
            # create kdtree for neighbor points
            neighbortree = spatial.KDTree(points_neighbor.numpy().tolist())
            for j in range( gt_feature_len ):
                fToTempDis = 0
                for it in range(points_neighbor.size()[0]):
                    ith = math.floor(abs(points_neighbor[it][0] + radius) / voxel_len)
                    jth = math.floor(abs(points_neighbor[it][1] + radius) / voxel_len)
                    kth = math.floor(abs(points_neighbor[it][2] + radius) / voxel_len)
                    # TODO 下标出错临时解决
                    ith = ith if ith < 30 else 29
                    jth = jth if jth < 30 else 29
                    kth = ith if kth < 30 else 29
                    iOneIdx = ith + jth*voxel_num_1dim + kth*voxel_num_1dim**2
                    fOneDis = vShapedicts[j][iOneIdx]
                    if fToTempDis < fOneDis:
                        fToTempDis = fOneDis

                fToSourceDis = 0
                for it in range(len(vKeyshapes[j])):
                    oneneardis, _idx = neighbortree.query(vKeyshapes[j][it],1)
                    if fToSourceDis < oneneardis:
                        fToSourceDis = oneneardis

                # print(fToSourceDis)
                fGenHdis = fToTempDis if fToTempDis > fToSourceDis else fToSourceDis
                fGenHdis = 1.0 if fGenHdis > radius else fGenHdis / radius
                feature[k, j, i] = 1 - fGenHdis
                # import pdb; pdb.set_trace()
                # print(fGenHdis)
            # print(feature[k,:,i])

    # print(feature)
    # import pdb; pdb.set_trace()
    return feature

"""
def thread_compute(points, keyclouds, grouped_xyz, batch_index, radius, feature, \
                   batch_size, point_num, theads_num, HausdorffOp, vKeyshapes, vShapedicts):
    step = int(batch_size/theads_num)
    for k in range(step):
        k = k + batch_index * step
        for i in range(point_num):
            points_neighbor = grouped_xyz[k, :, i, :].transpose(0,1).numpy().tolist()
            neighbortree = spatial.KDTree(points_neighbor)
            for j in range( gt_feature_len ):
                fToTempDis = HausdorffOp.HausdorffDict(vCloud = points_neighbor, vDisDict = vShapedicts[j])
                fToSourceDis = HausdorffOp.HausdorffDisMax(vKeyshapes[j], neighbortree)
                fGenHdis = HausdorffOp.GeneralHausDis(fToTempDis, fToSourceDis)
                feature[k, j, i] = 1 - fGenHdis
"""
