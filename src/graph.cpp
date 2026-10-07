#include "graphweft/graph.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
namespace graphweft {
namespace {
template<class T> std::vector<T> read_binary(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  auto bytes = f.tellg();
  if (bytes < 0 || bytes % sizeof(T)) throw std::runtime_error("invalid binary size: " + p.string());
  std::vector<T> a(size_t(bytes) / sizeof(T)); f.seekg(0);
  if (!a.empty()) f.read(reinterpret_cast<char*>(a.data()), bytes);
  if (!f) throw std::runtime_error("short read: " + p.string());
  return a;
}
uint32_t bits(float x) { uint32_t v; std::memcpy(&v, &x, 4); return v; }
void hash_bytes(uint64_t& h, const void* data, size_t n) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i=0;i<n;++i) { h ^= p[i]; h *= 1099511628211ULL; }
}
}
HostGraph HostGraph::from_edges(uint32_t n, const std::vector<uint32_t>& src,
    const std::vector<uint32_t>& dst, const std::vector<float>& w, bool directed) {
  if (!n || src.size()!=dst.size() || src.size()!=w.size()) throw std::invalid_argument("invalid edge arrays");
  struct Edge { uint32_t s,d,b; float w; };
  std::vector<Edge> edges; edges.reserve(src.size());
  for (size_t i=0;i<src.size();++i) {
    if (src[i]>=n || dst[i]>=n || !std::isfinite(w[i])) throw std::invalid_argument("invalid vertex or nonfinite weight");
    edges.push_back({src[i],dst[i],bits(w[i]),w[i]});
  }
  if (!directed) {
    std::vector<std::tuple<uint32_t,uint32_t,uint32_t>> forward, reverse;
    forward.reserve(edges.size()); reverse.reserve(edges.size());
    for (const auto& e:edges) { forward.emplace_back(e.s,e.d,e.b); reverse.emplace_back(e.d,e.s,e.b); }
    std::sort(forward.begin(),forward.end()); std::sort(reverse.begin(),reverse.end());
    if (forward!=reverse) throw std::invalid_argument("undirected graph requires matching reverse edges and exact weights");
  }
  HostGraph g; g.vertices=n; g.directed=directed; g.row.assign(size_t(n)+1,0);
  for (auto& e:edges) ++g.row[e.s+1];
  std::partial_sum(g.row.begin(),g.row.end(),g.row.begin());
  g.col.resize(edges.size()); g.weight.resize(edges.size());
  auto cursor=g.row;
  for (auto& e:edges) { auto j=cursor[e.s]++; g.col[j]=e.d; g.weight[j]=e.w; }
  if (directed) {
    g.incoming_row.assign(size_t(n)+1,0);
    for (auto& e:edges) ++g.incoming_row[e.d+1];
    std::partial_sum(g.incoming_row.begin(),g.incoming_row.end(),g.incoming_row.begin());
    g.incoming_col.resize(edges.size()); g.incoming_weight.resize(edges.size());
    cursor=g.incoming_row;
    for (auto& e:edges) { auto j=cursor[e.d]++; g.incoming_col[j]=e.s; g.incoming_weight[j]=e.w; }
  }
  uint64_t h=14695981039346656037ULL;
  hash_bytes(h,&n,sizeof(n)); hash_bytes(h,&directed,sizeof(directed));
  for (auto& e:edges) { hash_bytes(h,&e.s,4); hash_bytes(h,&e.d,4); hash_bytes(h,&e.b,4); }
  std::ostringstream os; os<<std::hex<<std::setw(16)<<std::setfill('0')<<h; g.identity=os.str();
  return g;
}
HostGraph HostGraph::load(const std::string& path, bool directed, bool legacy_int_weights) {
  namespace fs=std::filesystem;
  fs::path p(path); std::vector<uint32_t> src,dst; std::vector<float> w; uint32_t n=0;
  if (fs::is_directory(p)) {
    auto rowp=p/"csr_vlist.bin"; auto colp=p/"csr_elist.bin";
    auto cols=read_binary<int32_t>(colp);
    auto bytes=fs::file_size(rowp);
    std::vector<uint64_t> row;
    if (bytes==0) throw std::runtime_error("empty CSR offsets");
    // Existing puercgp datasets store int32 offsets; use int64 only when int32 does not fit a valid CSR.
    auto r32=read_binary<int32_t>(rowp);
    if (r32.size()>=2 && r32[0]==0 && r32.back()>=0 && size_t(r32.back())==cols.size())
      row.assign(r32.begin(),r32.end());
    else {
      auto r64=read_binary<uint64_t>(rowp);
      if (r64.size()<2 || r64[0]!=0 || r64.back()!=cols.size()) throw std::runtime_error("invalid CSR offsets");
      row=std::move(r64);
    }
    if (row.size()-1>std::numeric_limits<uint32_t>::max()) throw std::runtime_error("too many vertices");
    n=uint32_t(row.size()-1);
    auto wp=p/"csr_weightlist.bin";
    if (fs::exists(wp)) {
      if(legacy_int_weights){
        auto ints=read_binary<int32_t>(wp);w.reserve(ints.size());
        for(int32_t value:ints)w.push_back(static_cast<float>(value));
      }else w=read_binary<float>(wp);
    } else w.assign(cols.size(),1.f);
    if (w.size()!=cols.size()) throw std::runtime_error("CSR weight count mismatch");
    bool sorted_unique=true;
    for (uint32_t v=0;v<n;++v) {
      if (row[v]>row[v+1] || row[v+1]>cols.size()) throw std::runtime_error("invalid CSR row");
      uint32_t previous=0;bool have_previous=false;
      for(uint64_t j=row[v];j<row[v+1];++j){
        if(cols[j]<0 || uint32_t(cols[j])>=n)throw std::runtime_error("CSR destination out of range");
        if(have_previous && uint32_t(cols[j])<=previous)sorted_unique=false;
        previous=uint32_t(cols[j]);have_previous=true;
      }
    }
    // The campaign converter emits sorted, duplicate-free rows.  Adopt that
    // representation directly so billion-edge CSR inputs are not expanded to
    // src/dst tuples and sorted again merely to reconstruct the same CSR.
    if(sorted_unique){
      HostGraph g;g.vertices=n;g.directed=directed;g.row=std::move(row);
      g.col.reserve(cols.size());for(int32_t value:cols)g.col.push_back(uint32_t(value));g.weight=std::move(w);
      if(!directed){
        for(uint32_t u=0;u<n;++u)for(uint64_t e=g.row[u];e<g.row[u+1];++e){
          const uint32_t v=g.col[e];const auto first=g.col.begin()+g.row[v],last=g.col.begin()+g.row[v+1];
          const auto reverse=std::lower_bound(first,last,u);
          if(reverse==last || *reverse!=u || bits(g.weight[size_t(reverse-g.col.begin())])!=bits(g.weight[e]))
            throw std::invalid_argument("undirected graph requires matching reverse edges and exact weights");
        }
      }else{
        g.incoming_row.assign(size_t(n)+1,0);
        for(uint32_t v:g.col)++g.incoming_row[v+1];
        std::partial_sum(g.incoming_row.begin(),g.incoming_row.end(),g.incoming_row.begin());
        g.incoming_col.resize(g.col.size());g.incoming_weight.resize(g.weight.size());auto cursor=g.incoming_row;
        for(uint32_t u=0;u<n;++u)for(uint64_t e=g.row[u];e<g.row[u+1];++e){
          const auto index=cursor[g.col[e]]++;g.incoming_col[index]=u;g.incoming_weight[index]=g.weight[e];
        }
      }
      uint64_t h=14695981039346656037ULL;hash_bytes(h,&n,sizeof(n));hash_bytes(h,&directed,sizeof(directed));
      for(uint32_t u=0;u<n;++u)for(uint64_t e=g.row[u];e<g.row[u+1];++e){
        const uint32_t b=bits(g.weight[e]);hash_bytes(h,&u,4);hash_bytes(h,&g.col[e],4);hash_bytes(h,&b,4);
      }
      std::ostringstream os;os<<std::hex<<std::setw(16)<<std::setfill('0')<<h;g.identity=os.str();return g;
    }
    src.reserve(cols.size()); dst.reserve(cols.size());
    for (uint32_t v=0;v<n;++v)
      for (uint64_t j=row[v];j<row[v+1];++j) { src.push_back(v); dst.push_back(uint32_t(cols[j])); }
  } else {
    std::ifstream f(p); if (!f) throw std::runtime_error("cannot open graph: "+path);
    std::string line; uint32_t a,b; float weight;
    while (std::getline(f,line)) {
      if (line.empty() || line[0]=='#' || line[0]=='%') continue;
      std::istringstream in(line); if (!(in>>a>>b)) throw std::runtime_error("invalid edge line");
      if (!(in>>weight)) weight=1.f;
      src.push_back(a); dst.push_back(b); w.push_back(weight);
      n=std::max(n,std::max(a,b)+1);
    }
  }
  return from_edges(n,src,dst,w,directed);
}
}
