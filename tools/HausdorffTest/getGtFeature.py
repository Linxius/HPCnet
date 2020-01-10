import torch
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

def getGtFeature(points, keyclouds, grouped_xyz, nsample, radius):
    HausdorffOp = Haus.Hausdorff(radius,radius/15.0)
    root = "./HausdorffTest/shapes/" + str(radius)
    vKeyshapes, vShapedicts = LoadGivenShapes(root)
    gt_feature_len = len(vShapedicts)

    batch_size = keyclouds.size()[0]
    point_dim = keyclouds.size()[1]
    point_num = keyclouds.size()[2]
    feature = torch.zeros(batch_size, len(vKeyshapes), point_num,requires_grad=False)

    pool = multiprocessing.Pool(theads_num)

    for batch_index in range(theads_num):
        pool.apply_async(thread_compute, (points, keyclouds, grouped_xyz, batch_index, radius, feature, \
                                          batch_size, point_num, theads_num, HausdorffOp, vKeyshapes, vShapedicts))

    pool.close()
    pool.join()

    # return torch.cat([points, feature], 1 )
    # import pdb; pdb.set_trace()
    return feature

def thread_compute(points, keyclouds, grouped_xyz, batch_index, radius, feature, \
# def thread_compute(points, batch_index, radius, feature, \
                   batch_size, point_num, theads_num, HausdorffOp, vKeyshapes, vShapedicts):
    step = int(batch_size/theads_num)
    for k in range(step):
        k = k + batch_index * step
        for i in range(point_num):
            points_neighbor = grouped_xyz[k, :, i, :].transpose(0,1).numpy().tolist()
            neighbortree = spatial.KDTree(points_neighbor)
            for j in range( gt_feature_len ):
                #compute Hausdorff distance from source to template
                fToTempDis = HausdorffOp.HausdorffDict(vCloud = points_neighbor, vDisDict = vShapedicts[j])

                #compute Hausdorff distance from template to source
                fToSourceDis = HausdorffOp.HausdorffDisMax(vKeyshapes[j], neighbortree)

                #compute the general Hausdorff distance, which is a two-side measurement
                fGenHdis = HausdorffOp.GeneralHausDis(fToTempDis, fToSourceDis)

                # vResCheck.append(fGenHdis)
                feature[k, j, i] = 1 - fGenHdis
                # feature[k, j, i] = fGenHdis
                # print(1 - fGenHdis)
                # import pdb; pdb.set_trace()
