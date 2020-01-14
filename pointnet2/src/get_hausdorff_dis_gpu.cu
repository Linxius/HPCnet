#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "get_hausdorff_dis_gpu.h"
#include "cuda_utils.h"

#define gt_feature_len 42
#define voxel_num_1dim 31

__global__ void get_hausdorff_dis_kernel_fast(const float* __restrict__ whole_points,
                                            const float* __restrict__ keypoints,
                                            const float* __restrict__ neighbor_points,
                                            float* __restrict__ features, float radius,
                                            int batch_size, int point_dim, int point_num, int nsample,
                                            const float* __restrict__ prior_shapes,
                                            const float* __restrict__ dis_dicts,
                                            float voxel_len, cudaStream_t stream){
    // whole_points: B N C
    // keypoints: B N C
    // neighbor_points: B N nsample C
    // prior_shapes: Nshapes Npoints_per_shape Cxyz
    // dis_dicts: Nshapes N_hash_grid_per_shape Cxyz
    // output:
    //     features: batch_size Nshapes point_num



    float r2 = radius ** 2
    // for k in range(batch_size):
        //     clouds = points[k,:,:].transpose(0,1)
        //     # keypoints = keyclouds[k,:,:].transpose(0,1)

        //     for i in range(point_num):
        //         points_neighbor = grouped_xyz[k,:,i,:].transpose(0,1)

        //         for j in range( gt_feature_len ):
        //             fToTempDis = 0
        //             for it in range(len(points_neighbor)):
        //                 ith = math.floor(abs(points_neighbor[it][0] + radius) / voxel_len)
        //                 jth = math.floor(abs(points_neighbor[it][1] + radius) / voxel_len)
        //                 kth = math.floor(abs(points_neighbor[it][2] + radius) / voxel_len)
        //                 assert ith < 31 and jth < 31 and kth < 31
        //                 iOneIdx = int(ith + jth * voxel_num_1dim + kth * voxel_num_1dim * voxel_num_1dim)
        //                 fOneDis = vShapedicts[j][iOneIdx]
        //                 if fToTempDis < fOneDis:
        //                     fToTempDis = fOneDis

        //             fToSourceDis = 0
        //             for it in range(len(vKeyshapes[j])):
        //                 for iit in range(points_neighbor.size()[0]):
        //                     oneneardis = ((vKeyshapes[j][it][0]-points_neighbor[iit][0])**2 + \
        //                                     (vKeyshapes[j][it][0]-points_neighbor[iit][0])**2 + \
        //                                     (vKeyshapes[j][it][0]-points_neighbor[iit][0])**2)
        //                     if fToSourceDis < oneneardis:
        //                         fToSourceDis = oneneardis
        //             fToSourceDis = math.sqrt(fToSourceDis)

        //             fGenHdis = fToTempDis if fToTempDis > fToSourceDis else fToSourceDis
        //             fGenHdis = 1.0 if fGenHdis > radius else fGenHdis / radius
        //             feature[k, j, i] = 1 - fGenHdis
}

void get_hausdorff_dis_kernel_launcher_fast(const float* whole_points, const float* keypoints,
                                           const float*  neighbor_points,
                                           float* features, float radius,
                                           int batch_size, int point_dim, int point_num, int nsample,
                                           const float* prior_shapes, const float* dis_dicts,
                                            float voxel_len, cudaStream_t stream){
    // whole_points: B N C
    // keypoints: B N C
    // neighbor_points: B N nsample C
    // prior_shapes: Nshapes Npoints_per_shape Cxyz
    // dis_dicts: Nshapes N_hash_grid_per_shape Cxyz
    // output:
    //     features: batch_size point_num Nshapes

    cudaError_t err;

    // dim3 blocks(DIVUP(point_num, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(THREADS_PER_BLOCK);
    // ball_query_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    get_hausdorff_dis_kernel_fast<<<blocks, threads, 0, stream>>>(
        whole_points, keypoints, neighbor_points, features, radius, batch_size, point_num,
        point_num, nsample, prior_shapes, dis_dicts, voxel_len)

    cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
