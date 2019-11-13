import numpy as np
import torch
import kdtree
from threading import Thread
import time

gt_feature_len = 9 #gt_feature_len <= 61

#TODO make it run faster
def getGtFeature(points, radius=0.2):
    #temp_len = geo_template
    # points: torch.Size([32, 3, 2500])
    temp_len = 16
    batch_size = points.size()[0]
    point_dim = points.size()[1]
    point_num = points.size()[2]

    feature = torch.zeros(batch_size, gt_feature_len, point_num,requires_grad=False)

    r2 = radius**2
    #NOTE asdfa
    #DONE get neighbour points
    #DONE get template response
    for i in range(batch_size):
        # print(len(points[i,:,:].transpose(0,1).numpy().tolist()))
        # start = time.time()
        tree = kdtree.create(points[i,:,:].transpose(0,1).numpy().tolist())
        # end = time.time()
        # print("kdtree create time: %s seconds"%(end - start))
        for j in range(point_num):
            # start = time.time()
            point = points[i, :, j]
            points_within_radius = tree.search_nn_dist(point, radius)
            pwr_len = len(points_within_radius)
            thread_step = int(pwr_len / 11)
            # print(len(points_within_radius))
            # import pdb; pdb.set_trace()

            threads_pool = [
                Thread(target=compute_wrapper, args=( 1,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=( 2,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=( 3,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=( 4,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=( 5,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=( 6,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=( 7,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=( 8,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=( 9,thread_step, points_within_radius, i, j, feature)),
                Thread(target=compute_wrapper, args=(10,thread_step, points_within_radius, i, j, feature))
            ]

            for thread in threads_pool:
                thread.start()

            for thread in threads_pool:
                thread.join()

            for k in range(gt_feature_len):
                # print("i:%d k:%d j:%d" % (i, k, j))
                feature[i, k-1, j] /= (len(points_within_radius))
            # end = time.time()
            # print("for j time: %s seconds"%(end - start))
    return torch.cat([points, feature], 1 )

def compute_feature(points_within_radius, i, k, j, feature):
    point = points_within_radius[k]
    x = point[0]
    y = point[1]
    z = point[2]
    feature[i, 0, j] += np.abs( y ) # y = 0
    feature[i, 1, j] += np.abs( x ) # x = 0
    feature[i, 2, j] += np.abs( (x+y) / np.sqrt(x**2+y**2+z**2) ) # x + y = 0
    feature[i, 3, j] += np.abs( (x-y) / np.sqrt(x**2+y**2+z**2) ) # x - y = 0
    feature[i, 4, j] += np.abs( z ) # z = 0
    feature[i, 5, j] += np.abs( (x+z) / np.sqrt(x**2+y**2+z**2) ) # x + z = 0
    feature[i, 6, j] += np.abs( (x-z) / np.sqrt(x**2+y**2+z**2) ) # x - z = 0
    feature[i, 7, j] += np.abs( (z+y) / np.sqrt(x**2+y**2+z**2) ) # y + z = 0
    feature[i, 8, j] += np.abs( (z-y) / np.sqrt(x**2+y**2+z**2) ) # y - z = 0

def compute_wrapper(n, thread_step, points_within_radius, i, j, feature):
    for k in range(n*thread_step, (n+1)*thread_step):
        compute_feature(points_within_radius, i, k, j, feature)
