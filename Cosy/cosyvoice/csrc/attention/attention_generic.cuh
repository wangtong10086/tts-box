/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace vllm {

// A vector type to store Q, K, V elements.
template<typename T, int VEC_SIZE>
struct Vec {};

// A vector type to store FP32 accumulators.
template<typename T>
struct FloatVec {};

// Template vector operations.
template<typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template<typename T>
inline __device__ float sum(T v);

template<typename T>
inline __device__ T add(T a, T b);

template<typename T>
inline __device__ T vec_div_scalar(T a, T b);

template<typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template<typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

template<typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}


template <typename T>
struct ComputeSqrt {
    inline static __device__ T compute(float value) {
        static_assert(sizeof(T) == 0, "Unsupported type");
        return T(); 
    }

    inline static __device__ T compute(int value) {
        return compute(static_cast<float>(value));
    }
};

template <>
struct ComputeSqrt<__nv_bfloat16> {
    inline static __device__ __nv_bfloat16 compute(float value) {
        return __float2bfloat16(sqrtf(value));
    }

    inline static __device__ __nv_bfloat16 compute(int value) {
        return compute(static_cast<float>(value));
    }
};

template <>
struct ComputeSqrt<__half> {
    inline static __device__ __half compute(float value) {
        return __float2half(sqrtf(value));
    }

    inline static __device__ __half compute(int value) {
        return compute(static_cast<float>(value));
    }
};

template <>
struct ComputeSqrt<float> {
    inline static __device__ float compute(float value) {
        return sqrtf(value);
    }

    inline static __device__ float compute(int value) {
        return compute(static_cast<float>(value));
    }
};

template <>
struct ComputeSqrt<uint16_t> {
    inline static __device__ uint16_t compute(float value) {
        return static_cast<uint16_t>(sqrtf(value));
    }

    inline static __device__ uint16_t compute(int value) {
        return compute(static_cast<float>(value));
    }
};

} // namespace vllm
