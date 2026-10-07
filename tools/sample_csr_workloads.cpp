#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
namespace {
template<class T>std::vector<T> read(const fs::path& path){
  std::ifstream input(path,std::ios::binary|std::ios::ate);if(!input)throw std::runtime_error("cannot open "+path.string());
  const auto bytes=input.tellg();if(bytes<0 || bytes%sizeof(T))throw std::runtime_error("unaligned "+path.string());
  std::vector<T> values(size_t(bytes)/sizeof(T));input.seekg(0);
  if(!values.empty())input.read(reinterpret_cast<char*>(values.data()),bytes);
  if(!input)throw std::runtime_error("short read "+path.string());return values;
}
struct Dsu{
  std::vector<uint32_t> parent,size;
  explicit Dsu(uint32_t n):parent(n),size(n,1){std::iota(parent.begin(),parent.end(),0);}
  uint32_t find(uint32_t x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;}
  void join(uint32_t a,uint32_t b){a=find(a);b=find(b);if(a==b)return;if(size[a]<size[b])std::swap(a,b);parent[b]=a;size[a]+=size[b];}
};
void write_queries(const fs::path& path,const std::string& identity,uint32_t capacity,
                   const std::vector<uint32_t>& vertices,size_t begin,size_t count,uint64_t id_base,const std::string& note){
  std::ofstream out(path);if(!out)throw std::runtime_error("cannot write "+path.string());
  out<<"# graph_identity="<<identity<<"\n# capacity="<<capacity<<"\n# "<<note
     <<"\n# id,source,score,offset,feature_key,algorithm,reference_rounds\n";
  for(size_t i=0;i<count;++i)out<<id_base+i<<','<<vertices[begin+i]<<",0,0,0,1,0\n";
}
}
int main(int argc,char** argv){
  try{
    if(argc!=6)throw std::invalid_argument("usage: graphweft_sample_csr_workloads CSR OUTPUT IDENTITY CAPACITY SEED");
    const fs::path graph=argv[1],output=argv[2];const std::string identity=argv[3];
    const uint32_t capacity=uint32_t(std::stoul(argv[4]));const uint64_t seed=std::stoull(argv[5]);
    auto row=read<int32_t>(graph/"csr_vlist.bin");auto col=read<int32_t>(graph/"csr_elist.bin");
    if(row.size()<2 || row.front()!=0 || row.back()<0 || size_t(row.back())!=col.size())throw std::runtime_error("invalid CSR");
    const uint32_t vertices=uint32_t(row.size()-1);Dsu dsu(vertices);
    for(uint32_t u=0;u<vertices;++u)for(int32_t e=row[u];e<row[u+1];++e){
      if(col[e]<0 || uint32_t(col[e])>=vertices)throw std::runtime_error("destination out of range");
      dsu.join(u,uint32_t(col[e]));
    }
    uint32_t largest=UINT32_MAX;for(uint32_t v=0;v<vertices;++v)if(dsu.find(v)==v &&
      (largest==UINT32_MAX || dsu.size[v]>dsu.size[largest]))largest=v;
    if(largest==UINT32_MAX)throw std::runtime_error("CSR has no vertices");
    std::vector<uint32_t> eligible;
    for(uint32_t v=0;v<vertices;++v)if(row[v+1]>row[v] && dsu.find(v)==largest)eligible.push_back(v);
    constexpr size_t ordinary=1024,candidates=4096,calibration=1024,total=ordinary+candidates+calibration;
    if(eligible.size()<total)throw std::runtime_error("largest component has fewer than 6144 positive-degree vertices");
    std::mt19937_64 random(seed);std::shuffle(eligible.begin(),eligible.end(),random);fs::create_directories(output);
    write_queries(output/"sssp.csv",identity,capacity,eligible,0,ordinary,100000,
                  "largest-component uniform sample without replacement");
    write_queries(output/"sssp_candidates.csv",identity,capacity,eligible,ordinary,candidates,300000,
                  "disjoint candidate pool for measured natural-short selection");
    write_queries(output/"calibration_sssp.csv",identity,capacity,eligible,ordinary+candidates,calibration,500000,
                  "disjoint selector-training source pool");
    std::ofstream manifest(output/"manifest.json");manifest<<"{\n  \"schema\": 1,\n  \"N\": 1024,\n"
      <<"  \"M\": "<<capacity<<",\n  \"seed\": "<<seed<<",\n  \"graph_identity\": \""<<identity
      <<"\",\n  \"largest_component_vertices\": "<<dsu.size[largest]
      <<",\n  \"eligible_vertices\": "<<eligible.size()<<"\n}\n";
  }catch(const std::exception& error){std::cerr<<"sample_csr_workloads: "<<error.what()<<'\n';return 1;}
}
