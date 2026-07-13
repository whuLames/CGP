#pragma once

#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

namespace puercgp {
namespace detail {

inline void throw_if_driver_error(CUresult status, const char* operation) {
  if (status == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  throw std::runtime_error(std::string(operation) + ": " +
                           (name ? name : "unknown") + " (" +
                           (message ? message : "unknown") + ")");
}

class green_context {
 public:
  green_context(int device_ordinal, unsigned int requested_sms) {
    throw_if_driver_error(cuInit(0), "cuInit");
    throw_if_driver_error(cuDeviceGet(&device_, device_ordinal), "cuDeviceGet");

    CUcontext primary = nullptr;
    throw_if_driver_error(cuDevicePrimaryCtxRetain(&primary, device_),
                          "cuDevicePrimaryCtxRetain");
    primary_retained_ = true;
    throw_if_driver_error(cuCtxSetCurrent(primary), "cuCtxSetCurrent");

    CUdevResource all{};
    throw_if_driver_error(
        cuDeviceGetDevResource(device_, &all, CU_DEV_RESOURCE_TYPE_SM),
        "cuDeviceGetDevResource");
    unsigned int group_count = 1;
    CUdevResource selected{};
    CUdevResource remaining{};
    throw_if_driver_error(
        cuDevSmResourceSplitByCount(&selected, &group_count, &all, &remaining,
                                    0, requested_sms),
        "cuDevSmResourceSplitByCount");
    if (group_count != 1) {
      throw std::runtime_error("green context did not produce one SM group");
    }
    actual_sms_ = selected.sm.smCount;

    CUdevResourceDesc descriptor = nullptr;
    throw_if_driver_error(
        cuDevResourceGenerateDesc(&descriptor, &selected, 1),
        "cuDevResourceGenerateDesc");
    throw_if_driver_error(
        cuGreenCtxCreate(&context_, descriptor, device_,
                         CU_GREEN_CTX_DEFAULT_STREAM),
        "cuGreenCtxCreate");
    throw_if_driver_error(
        cuGreenCtxStreamCreate(&stream_, context_, CU_STREAM_NON_BLOCKING, 0),
        "cuGreenCtxStreamCreate");
  }

  green_context(const green_context&) = delete;
  green_context& operator=(const green_context&) = delete;

  ~green_context() {
    if (stream_) cuStreamDestroy(stream_);
    if (context_) cuGreenCtxDestroy(context_);
    if (primary_retained_) cuDevicePrimaryCtxRelease(device_);
  }

  cudaStream_t stream() const { return reinterpret_cast<cudaStream_t>(stream_); }
  unsigned int actual_sm_count() const { return actual_sms_; }

 private:
  CUdevice device_ = 0;
  CUgreenCtx context_ = nullptr;
  CUstream stream_ = nullptr;
  unsigned int actual_sms_ = 0;
  bool primary_retained_ = false;
};

class green_context_pair {
 public:
  green_context_pair(int device_ordinal, unsigned int first_sms) {
    throw_if_driver_error(cuInit(0), "cuInit");
    throw_if_driver_error(cuDeviceGet(&device_, device_ordinal), "cuDeviceGet");
    CUcontext primary = nullptr;
    throw_if_driver_error(cuDevicePrimaryCtxRetain(&primary, device_),
                          "cuDevicePrimaryCtxRetain");
    primary_retained_ = true;
    throw_if_driver_error(cuCtxSetCurrent(primary), "cuCtxSetCurrent");

    CUdevResource all{};
    throw_if_driver_error(
        cuDeviceGetDevResource(device_, &all, CU_DEV_RESOURCE_TYPE_SM),
        "cuDeviceGetDevResource");
    unsigned int group_count = 1;
    CUdevResource first{};
    CUdevResource second{};
    throw_if_driver_error(
        cuDevSmResourceSplitByCount(&first, &group_count, &all, &second, 0,
                                    first_sms),
        "cuDevSmResourceSplitByCount(pair)");
    if (group_count != 1 || second.sm.smCount == 0)
      throw std::runtime_error("green context pair requires two SM groups");
    first_sms_ = first.sm.smCount;
    second_sms_ = second.sm.smCount;

    create_context(first, &first_context_, &first_stream_);
    create_context(second, &second_context_, &second_stream_);
  }

  green_context_pair(const green_context_pair&) = delete;
  green_context_pair& operator=(const green_context_pair&) = delete;

  ~green_context_pair() {
    if (first_stream_) cuStreamDestroy(first_stream_);
    if (second_stream_) cuStreamDestroy(second_stream_);
    if (first_context_) cuGreenCtxDestroy(first_context_);
    if (second_context_) cuGreenCtxDestroy(second_context_);
    if (primary_retained_) cuDevicePrimaryCtxRelease(device_);
  }

  cudaStream_t first_stream() const {
    return reinterpret_cast<cudaStream_t>(first_stream_);
  }
  cudaStream_t second_stream() const {
    return reinterpret_cast<cudaStream_t>(second_stream_);
  }
  unsigned int first_sm_count() const { return first_sms_; }
  unsigned int second_sm_count() const { return second_sms_; }

 private:
  void create_context(CUdevResource& resource, CUgreenCtx* context,
                      CUstream* stream) {
    CUdevResourceDesc descriptor = nullptr;
    throw_if_driver_error(
        cuDevResourceGenerateDesc(&descriptor, &resource, 1),
        "cuDevResourceGenerateDesc(pair)");
    throw_if_driver_error(
        cuGreenCtxCreate(context, descriptor, device_,
                         CU_GREEN_CTX_DEFAULT_STREAM),
        "cuGreenCtxCreate(pair)");
    throw_if_driver_error(
        cuGreenCtxStreamCreate(stream, *context, CU_STREAM_NON_BLOCKING, 0),
        "cuGreenCtxStreamCreate(pair)");
  }

  CUdevice device_ = 0;
  CUgreenCtx first_context_ = nullptr;
  CUgreenCtx second_context_ = nullptr;
  CUstream first_stream_ = nullptr;
  CUstream second_stream_ = nullptr;
  unsigned int first_sms_ = 0;
  unsigned int second_sms_ = 0;
  bool primary_retained_ = false;
};

}  // namespace detail
}  // namespace puercgp
