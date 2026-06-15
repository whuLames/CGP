#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <puercgp/algorithms/hybrid.hxx>
#include <puercgp/core/types.hxx>

namespace puercgp {

// 单条 hybrid query 的描述符
//   source        : 起点 vertex id（WCC 不使用，可填 0）
//   kind          : 算法 tag，决定 update 段的 reduce 语义
//   source_value  : 起点初始值（BFS=0, SSSP=0；WCC 由专门 init kernel 设 vertex id）
struct query_descriptor_t {
  int source = 0;
  algorithms::algo_kind_t kind = algorithms::algo_kind_t::bfs;
  algorithms::unified_value_t source_value =
      algorithms::unified_source_value();
};

// 异构 query batch 容器：host 持有描述符列表，按需 upload 到 device
// 生命周期：upload_to_device() 返回的裸指针在 batch 对象存活期内有效
class hybrid_query_batch {
 public:
  hybrid_query_batch() = default;

  explicit hybrid_query_batch(std::vector<query_descriptor_t> descs)
      : descs_(std::move(descs)) {}

  std::size_t size() const noexcept { return descs_.size(); }
  bool empty() const noexcept { return descs_.empty(); }
  const std::vector<query_descriptor_t>& descriptors() const { return descs_; }
  const query_descriptor_t& operator[](std::size_t i) const {
    return descs_.at(i);
  }

  // 校验：非空、不超 max_queries（0 表示不限）、不超 64（query_mask_t 硬限）、
  //       source 非负
  void validate(std::size_t max_queries = 0) const {
    if (descs_.empty()) {
      throw std::invalid_argument("hybrid_query_batch must be non-empty");
    }
    if (max_queries != 0 && descs_.size() > max_queries) {
      throw std::invalid_argument("hybrid_query_batch exceeds max_queries");
    }
    if (descs_.size() > 64) {
      throw std::invalid_argument(
          "hybrid batch supports at most 64 queries (query_mask_t = uint64)");
    }
    for (const auto& d : descs_) {
      if (d.source < 0) {
        throw std::invalid_argument("query source must be non-negative");
      }
    }
  }

  // device 视图：传给 kernel 的三类裸指针
  struct device_views {
    const int* sources = nullptr;                          // [Q]
    const algorithms::algo_kind_t* kinds = nullptr;        // [Q]
    const algorithms::unified_value_t* source_values = nullptr;  // [Q]
  };

  // 上传到 device（同步）。返回的指针在本 batch 对象存活期内有效。
  // const 方法：device 缓存成员声明为 mutable，允许 const batch 调用
  device_views upload_to_device() const {
    const std::size_t q = descs_.size();
    std::vector<int> h_src(q);
    std::vector<algorithms::algo_kind_t> h_kind(q);
    std::vector<algorithms::unified_value_t> h_sv(q);
    for (std::size_t i = 0; i < q; ++i) {
      h_src[i] = descs_[i].source;
      h_kind[i] = descs_[i].kind;
      h_sv[i] = descs_[i].source_value;
    }
    d_sources_ = thrust::device_vector<int>(h_src);
    d_kinds_ = thrust::device_vector<algorithms::algo_kind_t>(h_kind);
    d_source_values_ =
        thrust::device_vector<algorithms::unified_value_t>(h_sv);
    return device_views{
        thrust::raw_pointer_cast(d_sources_.data()),
        thrust::raw_pointer_cast(d_kinds_.data()),
        thrust::raw_pointer_cast(d_source_values_.data())};
  }

  // 预算 BFS slot 的 mask（host 端，作为 kernel 参数传入）
  // 避免 kernel inner loop 内查 slot_kinds 表，O(1) 寄存器常量
  query_mask_t bfs_slot_mask() const {
    query_mask_t m = 0;
    for (std::size_t i = 0; i < descs_.size(); ++i) {
      if (descs_[i].kind == algorithms::algo_kind_t::bfs) {
        m |= (query_mask_t{1} << i);
      }
    }
    return m;
  }

  // 预算非 BFS slot 的 mask（SSSP + WCC）
  // 注意：显式遍历构造，不能用 ~bfs_slot_mask()（会反转高位引入虚假 slot）
  query_mask_t nonbfs_slot_mask() const {
    query_mask_t m = 0;
    for (std::size_t i = 0; i < descs_.size(); ++i) {
      if (descs_[i].kind != algorithms::algo_kind_t::bfs) {
        m |= (query_mask_t{1} << i);
      }
    }
    return m;
  }

  bool has_bfs() const {
    for (const auto& d : descs_) {
      if (d.kind == algorithms::algo_kind_t::bfs) return true;
    }
    return false;
  }

  bool has_wcc() const {
    for (const auto& d : descs_) {
      if (d.kind == algorithms::algo_kind_t::wcc) return true;
    }
    return false;
  }

 private:
  std::vector<query_descriptor_t> descs_;
  mutable thrust::device_vector<int> d_sources_;
  mutable thrust::device_vector<algorithms::algo_kind_t> d_kinds_;
  mutable thrust::device_vector<algorithms::unified_value_t> d_source_values_;
};

}  // namespace puercgp
