#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct csr_graph {
  std::vector<int> rows;
  std::vector<int> cols;
};

struct bfs_result {
  int eccentricity = 0;
  int farthest = -1;
  int reached = 0;
  std::uint64_t distance_sum = 0;
};

struct candidate {
  int source = -1;
  int estimate = 0;
  int eccentricity = 0;
  int reached = 0;
  std::uint64_t distance_sum = 0;
};

csr_graph load_csr(const std::string& directory) {
  namespace fs = std::filesystem;
  fs::path rows_path = fs::path(directory) / "csr_vlist.bin";
  fs::path cols_path = fs::path(directory) / "csr_elist.bin";
  if (!fs::is_regular_file(rows_path) || !fs::is_regular_file(cols_path))
    throw std::runtime_error("missing csr_vlist.bin or csr_elist.bin");
  if (fs::file_size(rows_path) % sizeof(int) != 0 ||
      fs::file_size(cols_path) % sizeof(int) != 0)
    throw std::runtime_error("selector requires int32 CSR");
  csr_graph graph;
  graph.rows.resize(fs::file_size(rows_path) / sizeof(int));
  graph.cols.resize(fs::file_size(cols_path) / sizeof(int));
  std::ifstream rows(rows_path, std::ios::binary);
  std::ifstream cols(cols_path, std::ios::binary);
  rows.read(reinterpret_cast<char*>(graph.rows.data()),
            graph.rows.size() * sizeof(int));
  cols.read(reinterpret_cast<char*>(graph.cols.data()),
            graph.cols.size() * sizeof(int));
  if (!rows || !cols) throw std::runtime_error("failed to read CSR");
  return graph;
}

class bfs_runner {
 public:
  explicit bfs_runner(const csr_graph& graph)
      : graph_(graph), distance_(graph.rows.size() - 1, -1) {
    queue_.reserve(distance_.size());
    touched_.reserve(distance_.size());
  }

  bfs_result run(int source, std::vector<int>* distance_output = nullptr) {
    for (int v : touched_) distance_[v] = -1;
    touched_.clear();
    queue_.clear();
    distance_[source] = 0;
    touched_.push_back(source);
    queue_.push_back(source);
    int farthest = source;
    std::uint64_t distance_sum = 0;
    for (std::size_t head = 0; head < queue_.size(); ++head) {
      int v = queue_[head];
      if (distance_[v] > distance_[farthest]) farthest = v;
      for (int edge = graph_.rows[v]; edge < graph_.rows[v + 1]; ++edge) {
        int neighbor = graph_.cols[edge];
        if (distance_[neighbor] != -1) continue;
        distance_[neighbor] = distance_[v] + 1;
        distance_sum += static_cast<std::uint64_t>(distance_[neighbor]);
        touched_.push_back(neighbor);
        queue_.push_back(neighbor);
      }
    }
    if (distance_output) *distance_output = distance_;
    return {distance_[farthest], farthest, static_cast<int>(queue_.size()),
            distance_sum};
  }

 private:
  const csr_graph& graph_;
  std::vector<int> distance_;
  std::vector<int> queue_;
  std::vector<int> touched_;
};

void write_workload(const std::filesystem::path& output,
                    const std::vector<candidate>& low,
                    const std::vector<candidate>& high, int count) {
  std::ofstream csv(output);
  csv << "query_id,source,eccentricity,reached,distance_sum,group\n";
  int half = count / 2;
  for (int i = 0; i < half; ++i) {
    const candidate& first = low[i];
    const candidate& second = high[i];
    csv << 2 * i << ',' << first.source << ',' << first.eccentricity << ','
        << first.reached << ',' << first.distance_sum << ",low\n";
    csv << 2 * i + 1 << ',' << second.source << ',' << second.eccentricity
        << ',' << second.reached << ',' << second.distance_sum << ",high\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: select_bfs_level_sources <csr-dir> <output-dir> "
                 "[candidate-count=2048]\n";
    return 2;
  }
  int candidate_count = argc > 3 ? std::stoi(argv[3]) : 2048;
  if (candidate_count < 800) candidate_count = 800;
  auto graph = load_csr(argv[1]);
  int V = static_cast<int>(graph.rows.size() - 1);
  bfs_runner runner(graph);

  auto seed_result = runner.run(0);
  std::vector<int> distance_a;
  auto from_a = runner.run(seed_result.farthest, &distance_a);
  std::vector<int> distance_b;
  auto from_b = runner.run(from_a.farthest, &distance_b);
  int giant_size = from_a.reached;
  std::cout << "V=" << V << " E=" << graph.cols.size()
            << " giant=" << giant_size << " landmark_a="
            << seed_result.farthest << " landmark_b=" << from_a.farthest
            << " double_sweep=" << from_a.eccentricity << '\n';

  std::vector<int> giant_vertices;
  giant_vertices.reserve(giant_size);
  for (int v = 0; v < V; ++v)
    if (distance_a[v] >= 0 && distance_b[v] >= 0) giant_vertices.push_back(v);
  std::sort(giant_vertices.begin(), giant_vertices.end(), [&](int x, int y) {
    int ex = std::max(distance_a[x], distance_b[x]);
    int ey = std::max(distance_a[y], distance_b[y]);
    if (ex != ey) return ex < ey;
    return x < y;
  });

  std::vector<candidate> candidates;
  candidates.reserve(candidate_count);
  for (int i = 0; i < candidate_count; ++i) {
    std::size_t rank = static_cast<std::size_t>(i) *
        (giant_vertices.size() - 1) / (candidate_count - 1);
    int source = giant_vertices[rank];
    auto exact = runner.run(source);
    candidates.push_back({source,
                          std::max(distance_a[source], distance_b[source]),
                          exact.eccentricity, exact.reached,
                          exact.distance_sum});
    if ((i + 1) % 128 == 0)
      std::cout << "validated_candidates=" << (i + 1) << '/'
                << candidate_count << '\n';
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& x,
                                                      const auto& y) {
    if (x.eccentricity != y.eccentricity)
      return x.eccentricity < y.eccentricity;
    return x.source < y.source;
  });

  std::vector<candidate> low(candidates.begin(), candidates.begin() + 400);
  std::vector<candidate> high(candidates.end() - 400, candidates.end());
  std::reverse(high.begin(), high.end());
  std::filesystem::path output = argv[2];
  std::filesystem::create_directories(output);
  std::ofstream all(output / "candidate_eccentricities.csv");
  all << "source,landmark_estimate,eccentricity,reached,distance_sum\n";
  for (const auto& item : candidates)
    all << item.source << ',' << item.estimate << ',' << item.eccentricity
        << ',' << item.reached << ',' << item.distance_sum << '\n';
  write_workload(output / "sources_n400.csv", low, high, 400);
  write_workload(output / "sources_n800.csv", low, high, 800);
  std::cout << "selected_low_range=" << low.front().eccentricity << '-'
            << low.back().eccentricity << " selected_high_range="
            << high.back().eccentricity << '-' << high.front().eccentricity
            << '\n';
  return 0;
}
