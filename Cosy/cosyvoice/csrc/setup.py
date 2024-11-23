from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

modules = [
    ('attention_cuda', ['attention.cpp', 'attention/attention_kernels.cu']),
    ('activation_cuda', ['activation.cpp', 'activation_kernels.cu']),
    ('cache_cuda', ['cache.cpp', 'cache_kernels.cu']),
    ('layernorm_cuda', ['layernorm.cpp', 'layernorm_kernels.cu']),
    ('pos_encoding_cuda', ['pos_encoding.cpp', 'pos_encoding_kernels.cu']),
]

ext_modules = [
    CUDAExtension(name, sources) for name, sources in modules
]

setup(
    name='kv_cache',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
