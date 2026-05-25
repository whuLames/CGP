// Convert Ligra Weighted Adjacency text format to GGR (Galois GR) binary format
// Usage: ./adjToGGR <input.adj> <output.gr>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
using namespace std;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <input.adj> <output.gr>" << endl;
    return 1;
  }

  ifstream in(argv[1]);
  if (!in.is_open()) {
    cout << "Cannot open " << argv[1] << endl;
    return 1;
  }

  string magic;
  in >> magic;
  bool weighted = (magic == "WeightedAdjacencyGraph");

  long n, m;
  in >> n >> m;

  // Read offsets (Ligra text format has exactly n offsets, not n+1)
  vector<int64_t> offsets(n + 1);
  for (long i = 0; i < n; i++) {
    in >> offsets[i];
  }
  offsets[n] = m; // sentinel

  // Read edge destinations
  vector<int32_t> edge_dst(m);
  for (long i = 0; i < m; i++) {
    in >> edge_dst[i];
  }

  // Read weights (if weighted)
  vector<int32_t> adjwgt;
  if (weighted) {
    adjwgt.resize(m);
    for (long i = 0; i < m; i++) {
      in >> adjwgt[i];
    }
  }

  in.close();

  // Write GGR binary
  ofstream out(argv[2], ios::out | ios::binary);
  if (!out.is_open()) {
    cout << "Cannot open " << argv[2] << " for writing" << endl;
    return 1;
  }

  // Header (32 bytes)
  uint64_t version = 1;
  uint64_t sizeEdgeTy = weighted ? sizeof(int32_t) : 0;
  uint64_t nvtxs = (uint64_t)n;
  uint64_t nedges = (uint64_t)m;
  out.write((char*)&version, sizeof(uint64_t));
  out.write((char*)&sizeEdgeTy, sizeof(uint64_t));
  out.write((char*)&nvtxs, sizeof(uint64_t));
  out.write((char*)&nedges, sizeof(uint64_t));

  // row_start[1..V] (skip offsets[0]=0)
  for (long i = 1; i <= n; i++) {
    int64_t val = (int64_t)offsets[i];
    out.write((char*)&val, sizeof(int64_t));
  }

  // edge_dst[E]
  out.write((char*)edge_dst.data(), sizeof(int32_t) * m);

  // Padding if weighted and E is odd
  if (weighted && (m % 2 == 1)) {
    int32_t pad = 0;
    out.write((char*)&pad, sizeof(int32_t));
  }

  // adjwgt[E] (weighted only)
  if (weighted) {
    out.write((char*)adjwgt.data(), sizeof(int32_t) * m);
  }

  out.close();
  cout << "Converted: " << n << " vertices, " << m << " edges" << endl;
  cout << "Weighted: " << (weighted ? "yes" : "no") << endl;
  cout << "Output: " << argv[2] << endl;
  return 0;
}
