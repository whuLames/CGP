#include <iostream>

#include <cuda_runtime.h>

#include <puercgp/core/green_context.hxx>

__global__ void increment_kernel(int* value) { atomicAdd(value, 1); }

int main(int argc, char** argv) {
  unsigned int requested = argc > 1 ? std::stoul(argv[1]) : 2;
  int* value = nullptr;
  cudaMallocManaged(&value, sizeof(int));
  *value = 0;
  {
    puercgp::detail::green_context context(0, requested);
    increment_kernel<<<context.actual_sm_count(), 32, 0, context.stream()>>>(
        value);
    cudaStreamSynchronize(context.stream());
    std::cout << "requested_sms=" << requested
              << " actual_sms=" << context.actual_sm_count()
              << " value=" << *value << '\n';
  }
  cudaFree(value);
  return 0;
}
