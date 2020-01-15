#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "get_hausdorff_dis_gpu.h"
#include "cuda_utils.h"

#define gt_num 42
#define voxel_dim 31

__global__ void get_hausdorff_dis_kernel_fast(const float* __restrict__ whole_points,
                                            const float* __restrict__ keypoints,
                                            const float* __restrict__ neighbor_points,
                                            float* __restrict__ features, float radius,
                                            int batch_size, int point_dim, int point_num, int nsample,
                                            const float* __restrict__ prior_shapes,
                                            const float* __restrict__ dis_dicts,
                                            float voxel_len, cudaStream_t stream){
    // whole_points: B N C
    // keypoints: B M C
    // neighbor_points: B M nsample C
    // prior_shapes: Nshapes Npoints_per_shape Cxyz
    // dis_dicts: Nshapes Ngrid Cxyz
    // output:
    //     features: batch_size Nshapes point_num

    // dim3 blocks(DIVUP(point_num, THREADS_PER_BLOCK), batch_size);  // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(DIVUP(THREADS_PER_BLOCK, gt_num), gt_num);
    int batch_idx_block = blockIdx.y;
    int point_idx_block = threadIdx.y;
    int gt_idx_block = threadIdx.x;
    // int point_idx = blockIdx.y * blockDim.x + threadIdx.y * threadDim.x
    int batch_idx = batch_idx_block;
    int point_idx = batch_idx * batch_size + point_idx_block * gt_num;
    int gt_idx = point_idx + gt_idx_block;


    float r2 = radius ** 2
    // for k in range(batch_size): # B parallelable
    //     for i in range(point_num): # M parallelable
    //         points_neighbor = grouped_xyz[k,:,i,:].transpose(0,1)
    //         for j in range( gt_feature_len ): # Nshapes parallelable
    //             fToTempDis = 0
    //             for it in range(len(points_neighbor)):
    //                 ith = math.floor(abs(points_neighbor[it][0] + radius) / voxel_len)
    //                 jth = math.floor(abs(points_neighbor[it][1] + radius) / voxel_len)
    //                 kth = math.floor(abs(points_neighbor[it][2] + radius) / voxel_len)
    //                 iOneIdx = int(ith + jth * voxel_num_1dim + kth * voxel_num_1dim * voxel_num_1dim)
    //                 fOneDis = vShapedicts[j][iOneIdx]
    //                 if fToTempDis < fOneDis:
    //                     fToTempDis = fOneDis

    //             fToSourceDis = 0
    //             for it in range(vKeyshapes[j].shape[0]):
    //                 minPointPairdis = 99.9
    //                 for iit in range(points_neighbor.size()[0]):
    //                     oneneardis = ((vKeyshapes[j][it][0]-points_neighbor[iit][0])**2 + \
    //                                     (vKeyshapes[j][it][1]-points_neighbor[iit][1])**2 + \
    //                                     (vKeyshapes[j][it][2]-points_neighbor[iit][2])**2)
    //                     if minPointPairdis > oneneardis:
    //                         minPointPairdis = oneneardis
    //                 if fToSourceDis < minPointPairdis:
    //                     fToSourceDis = minPointPairdis

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

    dim3 blocks(DIVUP(point_num, THREADS_PER_BLOCK), batch_size);
    dim3 threads(DIVUP(THREADS_PER_BLOCK, gt_num), gt_num);

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
