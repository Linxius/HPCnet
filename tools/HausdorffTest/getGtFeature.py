import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import math
# from scipy import spatial
from HausdorffTest.ReadShapes import read_keyfile
from HausdorffTest.ReadShapes import LoadGivenShapes
import time
import os
import sys
# import HausdorffTest.Hausdorff as Haus
# import sys
# sys.path.append('..')
# from pointnet2.pointnet2_utils import grouping_operation, ball_query
# gt_feature_len = 42

theads_num = 8

# def getGtFeature(points, keyclouds, grouped_xyz, nsample, radius):
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
        feature = torch.zeros(batch_size, len(vKeyshapes), point_num,requires_grad=False)

        whole_points_neighbor = grouped_xyz.permute(1, 2, 3, 0)

        r2 = radius ** 2
        for k in range(batch_size):
            # clouds = points[k,:,:].transpose(0,1).numpy().tolist()
            # keypoints = keyclouds[k,:,:].transpose(0,1).numpy().tolist()
            clouds = points[k,:,:].transpose(0,1)
            keypoints = keyclouds[k,:,:].transpose(0,1)
            # srctree = spatial.KDTree(data = clouds)

            for i in range(point_num):
                # neighidxs = srctree.query_ball_point(keypoints[i],radius)
                # # points_neighbor = HausdorffOp.RelativeCor(clouds, neighidxs, keypoints[i])
                # points_neighbor = [[0]*3 for row in range(len(neighidxs))]
                # for it in range(len(neighidxs)):
                #     points_neighbor[it][0] = clouds[neighidxs[it]][0] - keypoints[i][0]
                #     points_neighbor[it][1] = clouds[neighidxs[it]][1] - keypoints[i][1]
                #     points_neighbor[it][2] = clouds[neighidxs[it]][2] - keypoints[i][2]
                # points_neighbor = []
                # print(clouds.size())
                # for it in range( clouds.size()[1]):
                #     if r2 > ((clouds[it][0]-keypoints[i][0])**2+\
                #              (clouds[it][1]-keypoints[i][1])**2+\
                #              (clouds[it][2]-keypoints[i][2])**2):
                #         points_neighbor.append(clouds[i] - keypoints[i])

                # points_neighbor = grouped_xyz[k, :, i, :].transpose(0,1).numpy().tolist()
                # neighbortree = spatial.KDTree(points_neighbor)
                points_neighbor = whole_points_neighbor[k,:,i,:].transpose(0,1)

                for j in range( gt_feature_len ):
                    # fToTempDis = HausdorffOp.HausdorffDict(vCloud = points_neighbor, vDisDict = vShapedicts[j])
                    fToTempDis = -1.0*sys.float_info.max
                    #search the nearest contour point
                    for it in range(len(points_neighbor)):
                        # fOneDis = self.LocationtoDis(points_neighbor[it], vShapedicts)
                        ith = math.floor(abs(points_neighbor[it][0] + radius) / voxel_len)
                        jth = math.floor(abs(points_neighbor[it][1] + radius) / voxel_len)
                        kth = math.floor(abs(points_neighbor[it][2] + radius) / voxel_len)
                        # TODO 下标出错临时解决
                        ith = ith if ith < 30 else 29
                        jth = jth if jth < 30 else 29
                        kth = ith if kth < 30 else 29
                        # iOneIdx = self.Transfor3DTo1DIdx(ith, jth, kth, self.m_xnum, self.m_ynum)
                        iOneIdx = ith + jth * voxel_num_1dim + kth * voxel_num_1dim * voxel_num_1dim
                        fOneDis = vShapedicts[j][iOneIdx]
                        if fToTempDis < fOneDis:
                            fToTempDis = fOneDis

                    #compute Hausdorff distance from template to source
                    # fToSourceDis = HausdorffOp.HausdorffDisMax(vKeyshapes[j], neighbortree)
                    # fToSourceDis = -1.0*sys.float_info.max

                    fToSourceDis = 0
                    #search the nearest contour point
                    for it in range(len(vKeyshapes[j])):
                        # oneneardis, nearestidx = neighbortree.query(vKeyshapes[j][it],1)
                        # if fToSourceDis < oneneardis:
                        #     fToSourceDis = oneneardis
                        for iit in range(points_neighbor.size()[0]):
                            oneneardis = ((vKeyshapes[j][it][0]-points_neighbor[iit][0])**2 + \
                                          (vKeyshapes[j][it][0]-points_neighbor[iit][0])**2 + \
                                          (vKeyshapes[j][it][0]-points_neighbor[iit][0])**2)
                            if fToSourceDis < oneneardis:
                                fToSourceDis = oneneardis
                    fToSourceDis = math.sqrt(fToSourceDis)

                    #compute the general Hausdorff distance, which is a two-side measurement
                    # fGenHdis = HausdorffOp.GeneralHausDis(fToTempDis, fToSourceDis)
                    fGenHdis = fToTempDis if fToTempDis > fToSourceDis else fToSourceDis
                    fGenHdis = fGenHdis/radius
                    if fGenHdis > 1.0:
                        fGenHdis = 1.0
                    feature[k, j, i] = fGenHdis
                    # print(fGenHdis)
                    # import pdb; pdb.set_trace()
                # print(feature[k,:,i])

        # print(feature)
        # import pdb; pdb.set_trace()
        return feature
        # return torch.cat([points, feature], 1 )

    @staticmethod
    def backward(ctx, a = None):
        return None, None, None, None

get_gt_feature = getGtFeature.apply
