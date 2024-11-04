from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kv_cache',
    ext_modules=[
        CUDAExtension('attention_cuda', [
            'attention.cpp',
            'attention/attention_kernels.cu',
        ]),
        CUDAExtension('activation_cuda', [
            'activation.cpp',
            'activation_kernels.cu',
        ]),
        CUDAExtension('cache_cuda', [
            'cache.cpp',
            'cache_kernels.cu',
        ]),
        CUDAExtension('layernorm_cuda', [
            'layernorm.cpp',
            'layernorm_kernels.cu',
        ]),
        CUDAExtension('pos_encoding_cuda', [
            'pos_encoding.cpp',
            'pos_encoding_kernels.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
