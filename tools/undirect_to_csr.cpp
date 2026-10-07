#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs=std::filesystem;
namespace {
uint64_t pack(uint32_t source,uint32_t target){return uint64_t(source)<<32|target;}
uint32_t source(uint64_t edge){return uint32_t(edge>>32);}
uint32_t target(uint64_t edge){return uint32_t(edge);}
uint32_t weight(uint32_t a,uint32_t b,uint64_t seed){
  uint64_t x=(uint64_t(std::min(a,b))<<32)|std::max(a,b);
  x^=seed+0x9e3779b97f4a7c15ULL+(x<<6)+(x>>2);
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;
  return uint32_t(x%64)+1;
}
}
int main(int argc,char** argv){
  try{
    if(argc<3 || argc>6)throw std::invalid_argument(
      "usage: graphweft_undirect_to_csr INPUT OUTPUT [auto|zero|one] [vertices] [seed]");
    const fs::path input=argv[1],output=argv[2];
    const std::string indexing=argc>=4?argv[3]:"auto";
    if(indexing!="auto" && indexing!="zero" && indexing!="one")throw std::invalid_argument("invalid indexing mode");
    uint64_t declared_vertices=argc>=5?std::stoull(argv[4]):0;
    const uint64_t seed=argc>=6?std::stoull(argv[5]):20261007;
    if(fs::exists(output))throw std::invalid_argument("output already exists: "+output.string());
    std::ifstream in(input);if(!in)throw std::runtime_error("cannot open input: "+input.string());
    std::vector<uint64_t> edges;std::string line;bool matrix_market=false,dimensions_seen=false;
    uint64_t raw_edges=0,min_vertex=std::numeric_limits<uint64_t>::max(),max_vertex=0;
    while(std::getline(in,line)){
      if(line.empty())continue;
      if(line.rfind("%%MatrixMarket",0)==0){matrix_market=true;continue;}
      if(line[0]=='#' || line[0]=='%')continue;
      std::replace(line.begin(),line.end(),',',' ');
      std::istringstream row(line);uint64_t a=0,b=0,third=0;
      if(!(row>>a>>b))continue;
      if(matrix_market && !dimensions_seen){
        if(!(row>>third))throw std::runtime_error("invalid MatrixMarket dimensions");
        declared_vertices=std::max(a,b);dimensions_seen=true;
        if(third<5000000000ULL)edges.reserve(size_t(std::min<uint64_t>(2*third,1500000000ULL)));
        continue;
      }
      if(a>UINT32_MAX || b>UINT32_MAX)throw std::runtime_error("vertex ID exceeds uint32");
      edges.push_back(pack(uint32_t(a),uint32_t(b)));
      edges.push_back(pack(uint32_t(b),uint32_t(a)));
      min_vertex=std::min(min_vertex,std::min(a,b));max_vertex=std::max(max_vertex,std::max(a,b));++raw_edges;
    }
    if(!raw_edges)throw std::runtime_error("input contains no edges");
    const bool one_based=indexing=="one" || (indexing=="auto" && (matrix_market || min_vertex==1));
    if(one_based){
      if(min_vertex==0)throw std::runtime_error("one-based input contains vertex zero");
      for(auto& edge:edges)edge=pack(source(edge)-1,target(edge)-1);
      --max_vertex;
    }
    uint64_t vertices=declared_vertices?declared_vertices:max_vertex+1;
    bool compacted=false;std::vector<uint32_t> external_ids;
    const uint64_t external_id_span=max_vertex+1;
    if(vertices<=max_vertex){
      // Repository metadata normally reports the number of unique vertices, not
      // an upper bound for external IDs.  Compact a sparse external ID space
      // instead of rejecting an otherwise valid graph or allocating its holes.
      constexpr uint64_t max_remap_bytes=8ULL<<30;
      if(external_id_span>UINT32_MAX || external_id_span*sizeof(uint32_t)>max_remap_bytes)
        throw std::runtime_error("external vertex ID span is too large for dense remapping");
      std::vector<uint32_t> remap(size_t(external_id_span),UINT32_MAX);
      for(const auto edge:edges){remap[source(edge)]=0;remap[target(edge)]=0;}
      external_ids.reserve(size_t(declared_vertices));
      for(uint64_t id=0;id<external_id_span;++id)if(remap[size_t(id)]!=UINT32_MAX){
        remap[size_t(id)]=uint32_t(external_ids.size());
        external_ids.push_back(uint32_t(id)+(one_based?1U:0U));
      }
      for(auto& edge:edges)edge=pack(remap[source(edge)],remap[target(edge)]);
      vertices=external_ids.size();compacted=true;
    }
    if(!vertices || vertices>UINT32_MAX)throw std::runtime_error("invalid vertex count after ID handling");
    std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    if(edges.size()>=uint64_t(INT32_MAX))throw std::runtime_error("symmetric edge count exceeds legacy int32 CSR limit");
    fs::create_directories(output);
    std::ofstream rows(output/"csr_vlist.bin",std::ios::binary),cols(output/"csr_elist.bin",std::ios::binary),
      weights(output/"csr_weightlist.bin",std::ios::binary);
    if(!rows || !cols || !weights)throw std::runtime_error("cannot create CSR output");
    int32_t offset=0;rows.write(reinterpret_cast<const char*>(&offset),sizeof(offset));size_t cursor=0;
    for(uint32_t vertex=0;vertex<vertices;++vertex){
      while(cursor<edges.size() && source(edges[cursor])==vertex){
        const uint32_t destination=target(edges[cursor]);const int32_t col=int32_t(destination);
        const int32_t w=int32_t(weight(vertex,destination,seed));
        cols.write(reinterpret_cast<const char*>(&col),sizeof(col));
        weights.write(reinterpret_cast<const char*>(&w),sizeof(w));++cursor;++offset;
      }
      rows.write(reinterpret_cast<const char*>(&offset),sizeof(offset));
    }
    if(cursor!=edges.size() || !rows || !cols || !weights)throw std::runtime_error("CSR write failed");
    if(compacted){
      std::ofstream mapping(output/"external_vertex_ids.bin",std::ios::binary);
      mapping.write(reinterpret_cast<const char*>(external_ids.data()),
                    std::streamsize(external_ids.size()*sizeof(uint32_t)));
      if(!mapping)throw std::runtime_error("external vertex ID mapping write failed");
    }
    std::ofstream manifest(output/"conversion_manifest.json");
    manifest<<"{\n  \"schema\": 1,\n  \"source\": \""<<fs::absolute(input).string()<<"\",\n"
      <<"  \"indexing\": \""<<(one_based?"one":"zero")<<"\",\n  \"raw_edges\": "<<raw_edges
      <<",\n  \"declared_vertices\": "<<declared_vertices
      <<",\n  \"external_id_span\": "<<external_id_span
      <<",\n  \"external_ids_compacted\": "<<(compacted?"true":"false")
      <<",\n  \"external_vertex_map\": "<<(compacted?"\"external_vertex_ids.bin\"":"null")
      <<",\n  \"vertices\": "<<vertices<<",\n  \"symmetric_edges\": "<<edges.size()
      <<",\n  \"weight_seed\": "<<seed<<",\n  \"weight_range\": [1, 64],\n"
      <<"  \"duplicates_removed\": "<<(2*raw_edges-edges.size())<<"\n}\n";
    std::cout<<"vertices="<<vertices<<" raw_edges="<<raw_edges<<" symmetric_edges="<<edges.size()
      <<" indexing="<<(one_based?"one":"zero")<<" compacted="<<(compacted?"true":"false")<<'\n';
  }catch(const std::exception& error){std::cerr<<"undirect_to_csr: "<<error.what()<<'\n';return 1;}
}
