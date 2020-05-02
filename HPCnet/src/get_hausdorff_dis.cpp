#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "get_hausdorff_dis_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


int get_hausdorff_dis_wrapper_fast(at::Tensor neighbor_points_tensor,
                                   at::Tensor features_tensor, float radius,
                                   int batch_size, int whole_point_num,
                                   int keypoint_num, int neighbor_point_num,
                                   at::Tensor prior_points_tensor, at::Tensor dis_dicts_tensor,
                                   float voxel_len, int gt_num){
    CHECK_INPUT(neighbor_points_tensor);
    CHECK_INPUT(features_tensor);
    // CHECK_INPUT(prior_points_tensor);
    // CHECK_INPUT(dis_dicts_tensor);
    const float *neighbor_points = neighbor_points_tensor.data<float>();
    const float *prior_points = prior_points_tensor.data<float>();
    const float *dis_dicts = dis_dicts_tensor.data<float>();
    float *features = features_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    // ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx, stream);
    get_hausdorff_dis_kernel_launcher_fast(neighbor_points, features,
                                           radius, batch_size,
                                           whole_point_num, keypoint_num, neighbor_point_num,
                                           prior_points, dis_dicts, voxel_len, gt_num, stream);
    return 1;
}
