#pragma once

#include <stdexcept>
#include <string>
#include <vector>

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

// N-way SM partition. Each partition owns one green context and stream.
// The requested sum may be smaller than the device SM count.
class green_context_group {
 public:
  green_context_group(int device_ordinal,
                      const std::vector<unsigned int>& requested) {
    if (requested.empty())
      throw std::invalid_argument("green_context_group: empty partition list");
    throw_if_driver_error(cuInit(0), "cuInit");
    throw_if_driver_error(cuDeviceGet(&device_, device_ordinal), "cuDeviceGet");
    CUcontext primary = nullptr;
    throw_if_driver_error(cuDevicePrimaryCtxRetain(&primary, device_),
                          "cuDevicePrimaryCtxRetain");
    primary_retained_ = true;
    throw_if_driver_error(cuCtxSetCurrent(primary), "cuCtxSetCurrent");

    // The driver cannot split the returned remainder again. Create every
    // equal-sized partition in one split-by-count call.
    for (unsigned int size : requested) {
      if (size != requested.front())
        throw std::invalid_argument(
            "green_context_group: partitions must be uniform "
            "(single split-by-count call)");
    }
    CUdevResource pool{};
    throw_if_driver_error(
        cuDeviceGetDevResource(device_, &pool, CU_DEV_RESOURCE_TYPE_SM),
        "cuDeviceGetDevResource");
    unsigned int group_count = static_cast<unsigned int>(requested.size());
    std::vector<CUdevResource> selected(requested.size());
    CUdevResource remaining{};
    throw_if_driver_error(
        cuDevSmResourceSplitByCount(selected.data(), &group_count, &pool,
                                    &remaining, 0, requested.front()),
        "cuDevSmResourceSplitByCount(group)");
    if (group_count < requested.size())
      throw std::runtime_error(
          "green_context_group: driver produced fewer SM groups than "
          "requested");
    for (std::size_t g = 0; g < requested.size(); ++g) {
      if (selected[g].sm.smCount == 0)
        throw std::runtime_error("green_context_group: empty SM group");
      CUdevResourceDesc descriptor = nullptr;
      throw_if_driver_error(
          cuDevResourceGenerateDesc(&descriptor, &selected[g], 1),
          "cuDevResourceGenerateDesc(group)");
      CUgreenCtx context = nullptr;
      throw_if_driver_error(
          cuGreenCtxCreate(&context, descriptor, device_,
                           CU_GREEN_CTX_DEFAULT_STREAM),
          "cuGreenCtxCreate(group)");
      CUstream stream = nullptr;
      throw_if_driver_error(
          cuGreenCtxStreamCreate(&stream, context, CU_STREAM_NON_BLOCKING, 0),
          "cuGreenCtxStreamCreate(group)");
      contexts_.push_back(context);
      streams_.push_back(stream);
      sm_counts_.push_back(selected[g].sm.smCount);
    }
  }

  green_context_group(const green_context_group&) = delete;
  green_context_group& operator=(const green_context_group&) = delete;

  ~green_context_group() {
    for (CUstream stream : streams_)
      if (stream) cuStreamDestroy(stream);
    for (CUgreenCtx context : contexts_)
      if (context) cuGreenCtxDestroy(context);
    if (primary_retained_) cuDevicePrimaryCtxRelease(device_);
  }

  std::size_t size() const { return streams_.size(); }
  cudaStream_t stream(std::size_t i) const {
    return reinterpret_cast<cudaStream_t>(streams_.at(i));
  }
  unsigned int sm_count(std::size_t i) const { return sm_counts_.at(i); }

 private:
  CUdevice device_ = 0;
  std::vector<CUgreenCtx> contexts_;
  std::vector<CUstream> streams_;
  std::vector<unsigned int> sm_counts_;
  bool primary_retained_ = false;
};

}  // namespace detail
}  // namespace puercgp
