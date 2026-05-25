#pragma once

// #define CUDA_INCLUDE_TEMP  //注释掉，使用标准include路径

#ifdef CUDA_INCLUDE_TEMP

    #include </usr/local/cuda-11.7/include/cuda_runtime.h>
    #include </usr/local/cuda-11.7/include/cuda.h>
    #include </usr/local/cuda-11.7/include/device_launch_parameters.h>
#else
    #include <cuda_runtime.h>
    #include <cuda.h>
    #include <device_launch_parameters.h>
#endif
