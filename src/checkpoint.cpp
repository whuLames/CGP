#include "graphweft/checkpoint.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace graphweft {
namespace {
struct Header {
  char magic[8];
  char identity[17];
  uint32_t version,algorithm,layout,vertices,slots,physical_slots,group_width,words,frontier_count,round;
};
template<class T> void put(std::ofstream& f,const std::vector<T>& v){
  if(!v.empty())f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
}
template<class T> void get(std::ifstream& f,std::vector<T>& v){
  if(!v.empty())f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));
}
}
void save_checkpoint(const Checkpoint& c,const std::string& path){
  if(c.graph_identity.size()!=16 || c.old_values.size()!=uint64_t(c.vertices)*c.physical_slots ||
     c.frontier_mask.size()!=uint64_t(c.vertices)*c.words || c.frontier.size()!=c.frontier_count ||
     c.live_slots.size()!=c.slots || (!c.slot_algorithms.empty()&&c.slot_algorithms.size()!=c.slots))
    throw std::invalid_argument("invalid checkpoint dimensions");
  std::ofstream f(path,std::ios::binary|std::ios::trunc);
  if(!f)throw std::runtime_error("cannot create checkpoint: "+path);
  Header h{};std::memcpy(h.magic,"GWCP0002",8);std::memcpy(h.identity,c.graph_identity.data(),16);
  h.version=2;h.algorithm=uint32_t(c.algorithm);h.layout=uint32_t(c.layout);h.vertices=c.vertices;
  h.slots=c.slots;h.physical_slots=c.physical_slots;h.group_width=c.group_width;h.words=c.words;
  h.frontier_count=c.frontier_count;h.round=c.round;
  f.write(reinterpret_cast<const char*>(&h),sizeof(h));
  put(f,c.old_values);put(f,c.frontier_mask);put(f,c.frontier);put(f,c.live_slots);
  if(c.slot_algorithms.empty()){
    std::vector<Algorithm> algorithms(c.slots,c.algorithm);put(f,algorithms);
  }else put(f,c.slot_algorithms);
  if(!f)throw std::runtime_error("checkpoint write failed");
}
Checkpoint load_checkpoint(const std::string& path){
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open checkpoint: "+path);
  Header h{};f.read(reinterpret_cast<char*>(&h),sizeof(h));
  bool v1=!std::memcmp(h.magic,"GWCP0001",8)&&h.version==1;
  bool v2=!std::memcmp(h.magic,"GWCP0002",8)&&h.version==2;
  if(!f || (!v1&&!v2) || h.vertices==0 || h.slots==0 ||
     h.physical_slots<h.slots || h.group_width==0 || h.words!=(h.slots+63)/64 ||
     h.frontier_count>h.vertices || h.algorithm>2 || h.layout>1)throw std::runtime_error("invalid checkpoint header");
  Checkpoint c;c.graph_identity=std::string(h.identity,16);c.algorithm=Algorithm(h.algorithm);c.layout=Layout(h.layout);
  c.vertices=h.vertices;c.slots=h.slots;c.physical_slots=h.physical_slots;c.group_width=h.group_width;
  c.words=h.words;c.frontier_count=h.frontier_count;c.round=h.round;
  c.old_values.resize(uint64_t(c.vertices)*c.physical_slots);
  c.frontier_mask.resize(uint64_t(c.vertices)*c.words);
  c.frontier.resize(c.frontier_count);c.live_slots.resize(c.slots);
  get(f,c.old_values);get(f,c.frontier_mask);get(f,c.frontier);get(f,c.live_slots);
  c.slot_algorithms.resize(c.slots,c.algorithm);if(v2)get(f,c.slot_algorithms);
  if(!f || f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("invalid checkpoint payload");
  return c;
}
}
