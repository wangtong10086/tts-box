#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// 定义 vec_div_scalar 函数
inline __device__ __half vec_div_scalar(__half a, __half b) {
    return __hdiv(a, b);
}

inline __device__ __half2 vec_div_scalar(__half2 a, __half b) {
    __half2 c;
    c.x = vec_div_scalar(a.x, b);  
    c.y = vec_div_scalar(a.y, b);
    return c;
}

inline __device__ __half2 vec_div_scalar(__half2 a, uint16_t b) {
    __half2 c;
    __half b_half = __float2half(static_cast<float>(b));
    c.x = vec_div_scalar(a.x, b_half);  
    c.y = vec_div_scalar(a.y, b_half);
    return c;
}

inline __device__ __half vec_div_scalar(uint16_t a, uint16_t b) {
    __half a_half = __float2half(static_cast<float>(a));
    __half b_half = __float2half(static_cast<float>(b));
    return __hdiv(a_half, b_half);
}

inline __device__ uint32_t vec_div_scalar(uint32_t a, uint16_t b) {
    __half2 a_half2 = __halves2half2(__float2half(static_cast<float>(a & 0xFFFF)), __float2half(static_cast<float>((a >> 16) & 0xFFFF)));
    __half2 result_half2 = vec_div_scalar(a_half2, b);

    uint32_t result = (static_cast<uint16_t>(__half2float(result_half2.y)) << 16) | static_cast<uint16_t>(__half2float(result_half2.x));
    return result;
}

inline __device__ uint2 vec_div_scalar(uint2 a, uint16_t b) {
    uint2 c;
    c.x = vec_div_scalar(a.x, b);  
    c.y = vec_div_scalar(a.y, b);
    return c;
}

inline __device__ uint4 vec_div_scalar(uint4 a, uint16_t b) {
    uint4 c;
    c.x = vec_div_scalar(a.x, b);
    c.y = vec_div_scalar(a.y, b);
    c.z = vec_div_scalar(a.z, b);
    c.w = vec_div_scalar(a.w, b);
    return c;
}

// 内核函数
__global__ void test_vec_div_scalar_kernel(uint16_t* results, uint16_t a, uint16_t b) {
    uint16_t result1 = vec_div_scalar(a, b);
    uint2 result2 = vec_div_scalar(make_uint2(a, a + 1), b);
    uint4 result4 = vec_div_scalar(make_uint4(a, a + 1, a + 2, a + 3), b);

    results[0] = result1;
    results[1] = result2.x;
    results[2] = result2.y;
    results[3] = result4.x;
    results[4] = result4.y;
    results[5] = result4.z;
    results[6] = result4.w;
}

// 主函数
int main() {
    uint16_t a = 1024;
    uint16_t b = 16;

    uint16_t results[7];
    uint16_t* d_results;

    cudaMalloc((void**)&d_results, 7 * sizeof(uint16_t));
    test_vec_div_scalar_kernel<<<1, 1>>>(d_results, a, b);

    cudaMemcpy(results, d_results, 7 * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    std::cout << "Results:" << std::endl;
    std::cout << "vec_div_scalar(" << a << ", " << b << ") = " << results[0] << std::endl;
    std::cout << "vec_div_scalar(make_uint2(" << a << ", " << a + 1 << "), " << b << ") = (" << results[1] << ", " << results[2] << ")" << std::endl;
    std::cout << "vec_div_scalar(make_uint4(" << a << ", " << a + 1 << ", " << a + 2 << ", " << a + 3 << "), " << b << ") = (" << results[3] << ", " << results[4] << ", " << results[5] << ", " << results[6] << ")" << std::endl;

    cudaFree(d_results);

    return 0;
}