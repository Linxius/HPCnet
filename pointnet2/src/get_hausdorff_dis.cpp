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


int get_hausdorff_dis_wrapper_fast(at::Tensor whole_points_tensor, at::Tensor keypoints_tensor,
                                   at::Tensor neighbor_points_tensor,
                                   at::Tensor features_tensor, float radius,
                                   int batch_size, int point_dim, int point_num, int nsample,
                                   at::Tensor prior_shapes_tensor, at::Tensor dis_dicts_tensor,
                                   float voxel_len){
    CHECK_INPUT(whole_points_tensor);
    CHECK_INPUT(keypoints_tensor);
    CHECK_INPUT(neighbor_points_tensor);
    CHECK_INPUT(prior_shapes_tensor);
    CHECK_INPUT(dis_dicts_tensor);
    const float *whole_points = whole_points_tensor.data<float>();
    const float *keypoints = keypoints_tensor.data<float>();
    const float *neighbor_points = neighbor_points_tensor.data<float>();
    const float *prior_shapes = prior_shapes_tensor.data<float>();
    const float *dis_dicts = dis_dicts_tensor.data<float>();
    float *features = features_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    // ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx, stream);
    get_hausdorff_dis_kernel_launcher_fast(whole_points, keypoints, neighbor_points, features
                                           radius, batch_size, point_dim, point_num, nsample,
                                           prior_shapes, dis_dicts, voxel_len);
    return 1;
}
