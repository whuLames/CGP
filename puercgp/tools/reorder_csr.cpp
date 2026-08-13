#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

namespace fs = std::filesystem;

constexpr std::int32_t kReorderMagic = 0x52454f52;

struct options_t {
  fs::path input_dir;
  fs::path output_dir;
  fs::path mapping_input;
  fs::path secondary_mapping;
  fs::path mapping_output;
  fs::path segment_offsets_output;
  fs::path sources_path;
  std::string strategy;
  std::string within_level = "discovery";
  std::string neighbor_order = "preserve";
  std::string source_order = "input";
  std::string query_score = "borda";
  bool primary_min_distance = false;
  int query_score_bucket = 1;
  int local_degree_window = 0;
  int profile_group_size = 64;
  int phase_bits = 4;
  int query_count = 64;
  int hub_count = 4096;
  int hub_threshold = 0;
  int distance_bucket = 1;
  int level_bucket = 1;
  int threads = 0;
  unsigned int seed = 42;
};

struct csr_t {
  int vertices = 0;
  int edges = 0;
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<int> weights;
};

struct mapping_file_t {
  int strategy = 0;
  std::vector<int> new_to_old;
};

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

options_t parse_options(int argc, char **argv) {
  if (argc < 4) {
    throw std::invalid_argument(
        "Usage: reorder_csr <input-dir> <output-dir> <strategy> "
        "[--mapping-in=path] [--mapping-out=path] [--sources=path] "
        "[--segment-offsets-out=path] "
        "[--secondary-mapping=path] [--queries=64] "
        "[--within-level=discovery|degree-asc|degree-desc|mapping] "
        "[--neighbor-order=preserve|id|degree-asc|degree-desc] "
        "[--source-order=input|degree-asc|degree-desc|shuffle] "
        "[--query-score=borda|min|median|p75|max] "
        "[--query-score-bucket=1] "
        "[--primary-min-distance=0|1] "
        "[--local-degree-window=0] "
        "[--profile-group-size=64] [--phase-bits=4] "
        "[--distance-bucket=1] "
        "[--level-bucket=1] "
        "[--hub-count=4096] [--hub-threshold=0] [--threads=0] [--seed=42]\n"
        "Strategies: mapping, degree-asc, degree-desc, bucket-asc, "
        "bucket-desc, bfs, rcm, source-bfs, source-bfs-reverse, "
        "query-bfs-borda, query-bfs-borda-reverse, query-bfs-profile, "
        "query-coactivation, "
        "source-owner, source-sssp, "
        "hub-cluster, hub-sort, hub-group, minhash, random");
  }
  options_t options;
  options.input_dir = argv[1];
  options.output_dir = argv[2];
  options.strategy = argv[3];
  for (int i = 4; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value_after = [&](const std::string &prefix) {
      return argument.substr(prefix.size());
    };
    if (argument.rfind("--mapping-in=", 0) == 0)
      options.mapping_input = value_after("--mapping-in=");
    else if (argument.rfind("--secondary-mapping=", 0) == 0)
      options.secondary_mapping = value_after("--secondary-mapping=");
    else if (argument.rfind("--mapping-out=", 0) == 0)
      options.mapping_output = value_after("--mapping-out=");
    else if (argument.rfind("--segment-offsets-out=", 0) == 0)
      options.segment_offsets_output = value_after("--segment-offsets-out=");
    else if (argument.rfind("--sources=", 0) == 0)
      options.sources_path = value_after("--sources=");
    else if (argument.rfind("--queries=", 0) == 0)
      options.query_count = std::stoi(value_after("--queries="));
    else if (argument.rfind("--within-level=", 0) == 0)
      options.within_level = value_after("--within-level=");
    else if (argument.rfind("--neighbor-order=", 0) == 0)
      options.neighbor_order = value_after("--neighbor-order=");
    else if (argument.rfind("--source-order=", 0) == 0)
      options.source_order = value_after("--source-order=");
    else if (argument.rfind("--query-score=", 0) == 0)
      options.query_score = value_after("--query-score=");
    else if (argument.rfind("--query-score-bucket=", 0) == 0)
      options.query_score_bucket =
          std::stoi(value_after("--query-score-bucket="));
    else if (argument.rfind("--local-degree-window=", 0) == 0)
      options.local_degree_window =
          std::stoi(value_after("--local-degree-window="));
    else if (argument.rfind("--profile-group-size=", 0) == 0)
      options.profile_group_size =
          std::stoi(value_after("--profile-group-size="));
    else if (argument.rfind("--phase-bits=", 0) == 0)
      options.phase_bits = std::stoi(value_after("--phase-bits="));
    else if (argument.rfind("--primary-min-distance=", 0) == 0) {
      const auto value = value_after("--primary-min-distance=");
      if (value != "0" && value != "1") {
        throw std::invalid_argument("--primary-min-distance must be 0 or 1");
      }
      options.primary_min_distance = value == "1";
    } else if (argument.rfind("--hub-count=", 0) == 0)
      options.hub_count = std::stoi(value_after("--hub-count="));
    else if (argument.rfind("--hub-threshold=", 0) == 0)
      options.hub_threshold = std::stoi(value_after("--hub-threshold="));
    else if (argument.rfind("--distance-bucket=", 0) == 0)
      options.distance_bucket = std::stoi(value_after("--distance-bucket="));
    else if (argument.rfind("--level-bucket=", 0) == 0)
      options.level_bucket = std::stoi(value_after("--level-bucket="));
    else if (argument.rfind("--threads=", 0) == 0)
      options.threads = std::stoi(value_after("--threads="));
    else if (argument.rfind("--seed=", 0) == 0)
      options.seed =
          static_cast<unsigned int>(std::stoul(value_after("--seed=")));
    else
      throw std::invalid_argument("unknown option: " + argument);
  }
  const std::vector<std::string> strategies = {"mapping",
                                               "degree-asc",
                                               "degree-desc",
                                               "bucket-asc",
                                               "bucket-desc",
                                               "bfs",
                                               "rcm",
                                               "source-bfs",
                                               "source-bfs-reverse",
                                               "query-bfs-borda",
                                               "query-bfs-borda-reverse",
                                               "query-bfs-profile",
                                               "query-coactivation",
                                               "source-owner",
                                               "source-sssp",
                                               "hub-cluster",
                                               "hub-sort",
                                               "hub-group",
                                               "minhash",
                                               "random"};
  if (std::find(strategies.begin(), strategies.end(), options.strategy) ==
      strategies.end()) {
    throw std::invalid_argument("unsupported strategy: " + options.strategy);
  }
  if (options.strategy == "mapping" && options.mapping_input.empty()) {
    throw std::invalid_argument("mapping strategy requires --mapping-in");
  }
  if ((options.strategy == "source-bfs" ||
       options.strategy == "source-bfs-reverse" ||
       options.strategy == "query-bfs-borda" ||
       options.strategy == "query-bfs-borda-reverse" ||
       options.strategy == "query-bfs-profile" ||
       options.strategy == "query-coactivation" ||
       options.strategy == "source-owner" ||
       options.strategy == "source-sssp") &&
      options.sources_path.empty()) {
    throw std::invalid_argument("source-aware strategies require --sources");
  }
  if (options.query_count <= 0 || options.hub_count <= 0 ||
      options.hub_threshold < 0 || options.distance_bucket <= 0 ||
      options.level_bucket <= 0 || options.query_score_bucket <= 0 ||
      options.local_degree_window < 0 || options.profile_group_size <= 0 ||
      options.profile_group_size > 64 || options.phase_bits <= 0 ||
      options.phase_bits > 8 || options.threads < 0) {
    throw std::invalid_argument("invalid numeric option");
  }
  if (options.within_level != "discovery" &&
      options.within_level != "degree-asc" &&
      options.within_level != "degree-desc" &&
      options.within_level != "mapping") {
    throw std::invalid_argument("invalid --within-level value");
  }
  if (options.within_level == "mapping" && options.secondary_mapping.empty()) {
    throw std::invalid_argument(
        "--within-level=mapping requires --secondary-mapping");
  }
  if (options.neighbor_order != "preserve" && options.neighbor_order != "id" &&
      options.neighbor_order != "degree-asc" &&
      options.neighbor_order != "degree-desc") {
    throw std::invalid_argument("invalid --neighbor-order value");
  }
  if (options.source_order != "input" && options.source_order != "degree-asc" &&
      options.source_order != "degree-desc" &&
      options.source_order != "shuffle") {
    throw std::invalid_argument("invalid --source-order value");
  }
  if (options.query_score != "borda" && options.query_score != "min" &&
      options.query_score != "median" && options.query_score != "p75" &&
      options.query_score != "max") {
    throw std::invalid_argument("invalid --query-score value");
  }
  const int profile_dimensions =
      (options.query_count + options.profile_group_size - 1) /
      options.profile_group_size;
  if (options.strategy == "query-coactivation" &&
      profile_dimensions * options.phase_bits *
              (options.primary_min_distance ? 2 : 1) >
          64) {
    throw std::invalid_argument(
        "query-coactivation profile requires dimensions * phase bits <= 64");
  }
  if ((options.strategy == "query-bfs-borda" ||
       options.strategy == "query-bfs-borda-reverse" ||
       options.strategy == "query-bfs-profile") &&
      options.query_count > 64) {
    throw std::invalid_argument(
        "query-bfs strategies support at most 64 queries");
  }
  if (options.mapping_output.empty()) {
    options.mapping_output = options.output_dir / "reorder.map.bin";
  }
  if (!options.segment_offsets_output.empty() &&
      options.strategy != "source-sssp" &&
      options.strategy != "source-bfs" &&
      options.strategy != "source-bfs-reverse") {
    throw std::invalid_argument(
        "--segment-offsets-out requires a source BFS/SSSP strategy");
  }
  return options;
}

template <typename value_t>
std::vector<value_t> read_binary_vector(const fs::path &path,
                                        bool required = true) {
  if (!fs::is_regular_file(path)) {
    if (!required)
      return {};
    throw std::runtime_error("missing input file: " + path.string());
  }
  const auto bytes = fs::file_size(path);
  if (bytes % sizeof(value_t) != 0) {
    throw std::runtime_error("invalid binary file size: " + path.string());
  }
  std::vector<value_t> values(
      static_cast<std::size_t>(bytes / sizeof(value_t)));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(bytes));
  if (!input)
    throw std::runtime_error("failed to read: " + path.string());
  return values;
}

template <typename value_t>
void write_binary_vector(const fs::path &path,
                         const std::vector<value_t> &values) {
  const auto temporary = fs::path(path.string() + ".tmp");
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create: " + temporary.string());
  }
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(value_t)));
  output.close();
  if (!output)
    throw std::runtime_error("failed to write: " + path.string());
  fs::rename(temporary, path);
}

csr_t load_csr(const fs::path &directory) {
  csr_t graph;
  auto start = std::chrono::steady_clock::now();
  graph.row_offsets = read_binary_vector<int>(directory / "csr_vlist.bin");
  graph.columns = read_binary_vector<int>(directory / "csr_elist.bin");
  graph.weights =
      read_binary_vector<int>(directory / "csr_weightlist.bin", false);
  if (graph.row_offsets.size() < 2 ||
      graph.row_offsets.size() - 1 >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      graph.columns.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("graph exceeds 32-bit CSR limits");
  }
  graph.vertices = static_cast<int>(graph.row_offsets.size() - 1);
  graph.edges = static_cast<int>(graph.columns.size());
  if (graph.row_offsets.front() != 0 ||
      graph.row_offsets.back() != graph.edges ||
      (!graph.weights.empty() &&
       graph.weights.size() != graph.columns.size())) {
    throw std::runtime_error("inconsistent CSR input");
  }
  std::cout << "load_ms=" << elapsed_ms(start) << " vertices=" << graph.vertices
            << " edges=" << graph.edges
            << " weighted=" << (!graph.weights.empty() ? 1 : 0) << '\n';
  return graph;
}

std::vector<int> load_sources(const fs::path &path, int count, int vertices) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open sources file: " + path.string());
  }
  std::vector<int> sources;
  std::string line;
  while (std::getline(input, line) &&
         static_cast<int>(sources.size()) < count) {
    if (line.empty() || line[0] == '#')
      continue;
    std::stringstream row(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(row, field, ','))
      fields.push_back(field);
    try {
      const int source = std::stoi(fields.size() >= 2 ? fields[1] : fields[0]);
      if (source < 0 || source >= vertices) {
        throw std::runtime_error("source outside graph");
      }
      sources.push_back(source);
    } catch (const std::invalid_argument &) {
      if (sources.empty())
        continue;
      throw;
    }
  }
  if (static_cast<int>(sources.size()) != count) {
    throw std::runtime_error("sources file has fewer entries than requested");
  }
  return sources;
}

mapping_file_t load_mapping(const fs::path &path, int vertices) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open mapping: " + path.string());
  }
  std::int32_t magic = 0;
  std::int32_t count = 0;
  std::int32_t strategy = 0;
  input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  input.read(reinterpret_cast<char *>(&strategy), sizeof(strategy));
  if (!input || magic != kReorderMagic || count != vertices) {
    throw std::runtime_error("mapping header does not match input graph");
  }
  mapping_file_t mapping;
  mapping.strategy = strategy;
  mapping.new_to_old.resize(static_cast<std::size_t>(vertices));
  input.read(
      reinterpret_cast<char *>(mapping.new_to_old.data()),
      static_cast<std::streamsize>(mapping.new_to_old.size() * sizeof(int)));
  if (!input)
    throw std::runtime_error("truncated mapping file");
  return mapping;
}

void write_mapping(const fs::path &path, int strategy,
                   const std::vector<int> &new_to_old) {
  const auto temporary = fs::path(path.string() + ".tmp");
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  const std::int32_t magic = kReorderMagic;
  const std::int32_t count = static_cast<std::int32_t>(new_to_old.size());
  const std::int32_t tag = strategy;
  output.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  output.write(reinterpret_cast<const char *>(&count), sizeof(count));
  output.write(reinterpret_cast<const char *>(&tag), sizeof(tag));
  output.write(reinterpret_cast<const char *>(new_to_old.data()),
               static_cast<std::streamsize>(new_to_old.size() * sizeof(int)));
  output.close();
  if (!output)
    throw std::runtime_error("failed to write mapping");
  fs::rename(temporary, path);
}

int degree_bucket(int degree) {
  if (degree <= 0)
    return 0;
  int bucket = 0;
  while (degree > 1) {
    degree >>= 1;
    ++bucket;
  }
  return bucket + 1;
}

std::vector<int> degree_order(const csr_t &graph, bool descending) {
  std::vector<int> order(static_cast<std::size_t>(graph.vertices));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    const int left_degree =
        graph.row_offsets[left + 1] - graph.row_offsets[left];
    const int right_degree =
        graph.row_offsets[right + 1] - graph.row_offsets[right];
    if (left_degree != right_degree) {
      return descending ? left_degree > right_degree
                        : left_degree < right_degree;
    }
    return left < right;
  });
  return order;
}

std::vector<int> bucket_order(const csr_t &graph, bool descending) {
  constexpr int kBuckets = 32;
  std::vector<std::size_t> counts(kBuckets, 0);
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const int degree =
        graph.row_offsets[vertex + 1] - graph.row_offsets[vertex];
    ++counts[degree_bucket(degree)];
  }
  std::vector<std::size_t> positions(kBuckets, 0);
  std::size_t offset = 0;
  for (int index = 0; index < kBuckets; ++index) {
    const int bucket = descending ? (kBuckets - 1 - index) : index;
    positions[bucket] = offset;
    offset += counts[bucket];
  }
  std::vector<int> order(static_cast<std::size_t>(graph.vertices));
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const int degree =
        graph.row_offsets[vertex + 1] - graph.row_offsets[vertex];
    order[positions[degree_bucket(degree)]++] = vertex;
  }
  return order;
}

std::vector<int> bfs_order(const csr_t &graph, const std::vector<int> &seeds,
                           const std::string &within_level, bool reverse_order,
                           const std::vector<int> *secondary_order = nullptr,
                           int level_bucket = 1,
                           std::vector<std::size_t> *segment_offsets = nullptr) {
  std::vector<int> distance(static_cast<std::size_t>(graph.vertices), -1);
  std::vector<int> queue;
  queue.reserve(static_cast<std::size_t>(graph.vertices));
  for (int source : seeds) {
    if (distance[static_cast<std::size_t>(source)] == -1) {
      distance[static_cast<std::size_t>(source)] = 0;
      queue.push_back(source);
    }
  }
  std::size_t head = 0;
  std::vector<std::size_t> level_offsets{0, queue.size()};
  int current_level = 0;
  while (head < queue.size()) {
    const std::size_t level_end = level_offsets.back();
    while (head < level_end) {
      const int vertex = queue[head++];
      for (int edge = graph.row_offsets[vertex];
           edge < graph.row_offsets[vertex + 1]; ++edge) {
        const int neighbor = graph.columns[static_cast<std::size_t>(edge)];
        if (distance[static_cast<std::size_t>(neighbor)] == -1) {
          distance[static_cast<std::size_t>(neighbor)] = current_level + 1;
          queue.push_back(neighbor);
        }
      }
    }
    if (head < queue.size())
      level_offsets.push_back(queue.size());
    ++current_level;
  }

  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    if (distance[static_cast<std::size_t>(vertex)] == -1) {
      queue.push_back(vertex);
    }
  }
  if (level_offsets.back() != queue.size())
    level_offsets.push_back(queue.size());

  if (within_level == "mapping") {
    // Stable bucketization is equivalent to sorting each level by mapping rank,
    // but avoids an O(V log V) comparison sort on large, shallow graphs.
    const auto level_count = level_offsets.size() - 1;
    const auto bucket_count =
        (level_count + static_cast<std::size_t>(level_bucket) - 1) /
        static_cast<std::size_t>(level_bucket);
    std::vector<std::uint32_t> bucket_by_vertex(
        static_cast<std::size_t>(graph.vertices));
    std::vector<std::size_t> offsets(bucket_count + 1, 0);
    std::size_t bucket = 0;
    for (std::size_t level = 0; level < level_count;
         level += static_cast<std::size_t>(level_bucket), ++bucket) {
      const auto end_level =
          std::min(level + static_cast<std::size_t>(level_bucket),
                   level_count);
      const auto begin = level_offsets[level];
      const auto end = level_offsets[end_level];
      offsets[bucket + 1] = end - begin;
      for (auto position = begin; position < end; ++position) {
        bucket_by_vertex[static_cast<std::size_t>(queue[position])] =
            static_cast<std::uint32_t>(bucket);
      }
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    auto positions = offsets;
    std::vector<int> reordered(queue.size());
    for (int vertex : *secondary_order) {
      const auto vertex_bucket =
          bucket_by_vertex[static_cast<std::size_t>(vertex)];
      reordered[positions[vertex_bucket]++] = vertex;
    }
    queue.swap(reordered);
  } else if (within_level != "discovery") {
    for (std::size_t level = 0; level + 1 < level_offsets.size();
         level += static_cast<std::size_t>(level_bucket)) {
      const auto end_level =
          std::min(level + static_cast<std::size_t>(level_bucket),
                   level_offsets.size() - 1);
      const auto begin =
          queue.begin() + static_cast<std::ptrdiff_t>(level_offsets[level]);
      const auto end =
          queue.begin() + static_cast<std::ptrdiff_t>(level_offsets[end_level]);
      std::sort(begin, end, [&](int left, int right) {
        const int left_degree =
            graph.row_offsets[left + 1] - graph.row_offsets[left];
        const int right_degree =
            graph.row_offsets[right + 1] - graph.row_offsets[right];
        if (left_degree != right_degree) {
          return within_level == "degree-desc" ? left_degree > right_degree
                                               : left_degree < right_degree;
        }
        return left < right;
      });
    }
  }
  if (reverse_order)
    std::reverse(queue.begin(), queue.end());
  if (segment_offsets != nullptr) {
    segment_offsets->clear();
    if (!reverse_order) {
      for (std::size_t level = 0; level + 1 < level_offsets.size();
           level += static_cast<std::size_t>(level_bucket)) {
        segment_offsets->push_back(level_offsets[level]);
      }
      segment_offsets->push_back(queue.size());
    } else {
      segment_offsets->push_back(0);
      for (std::size_t end_level = level_offsets.size() - 1; end_level > 0;) {
        const std::size_t begin_level =
            end_level > static_cast<std::size_t>(level_bucket)
                ? end_level - static_cast<std::size_t>(level_bucket)
                : 0;
        segment_offsets->push_back(
            segment_offsets->back() + level_offsets[end_level] -
            level_offsets[begin_level]);
        end_level = begin_level;
      }
    }
  }
  std::cout << "bfs_levels=" << current_level
            << " reached=" << level_offsets[level_offsets.size() - 2]
            << " seeds=" << seeds.size() << " level_bucket=" << level_bucket
            << '\n';
  return queue;
}

std::vector<int> query_bfs_borda_order(const csr_t &graph,
                                       const std::vector<int> &sources,
                                       const std::vector<int> *secondary_order,
                                       bool reverse_score,
                                       const std::string &score_mode = "borda",
                                       bool primary_min_distance = false,
                                       int score_bucket = 1) {
  using mask_t = std::uint64_t;
  const auto vertex_count = static_cast<std::size_t>(graph.vertices);
  std::vector<mask_t> frontier(vertex_count, 0);
  std::vector<mask_t> next_frontier(vertex_count, 0);
  std::vector<mask_t> visited(vertex_count, 0);
  std::vector<std::uint32_t> distance_sums(vertex_count, 0);
  std::vector<unsigned char> reached_counts(vertex_count, 0);
  std::vector<unsigned char> minimum_distances(vertex_count, 255);
  std::vector<unsigned char> median_distances(vertex_count, 255);
  std::vector<unsigned char> p75_distances(vertex_count, 255);
  std::vector<unsigned char> maximum_distances(vertex_count, 0);
  const auto median_target =
      static_cast<unsigned int>((sources.size() + 1) / 2);
  const auto p75_target =
      static_cast<unsigned int>((3 * sources.size() + 3) / 4);

  for (std::size_t query = 0; query < sources.size(); ++query) {
    const auto bit = mask_t{1} << query;
    const auto source = static_cast<std::size_t>(sources[query]);
    frontier[source] |= bit;
    visited[source] |= bit;
    ++reached_counts[source];
    minimum_distances[source] = 0;
  }

  int level = 0;
  std::uint64_t reached_pairs = sources.size();
  while (true) {
    std::uint64_t next_pairs = 0;
    std::uint64_t next_vertices = 0;
#pragma omp parallel for reduction(+ : next_pairs, next_vertices) schedule(dynamic, 1024)
    for (int vertex = 0; vertex < graph.vertices; ++vertex) {
      mask_t candidates = 0;
      for (int edge = graph.row_offsets[vertex];
           edge < graph.row_offsets[vertex + 1]; ++edge) {
        candidates |= frontier[static_cast<std::size_t>(
            graph.columns[static_cast<std::size_t>(edge)])];
      }
      const auto index = static_cast<std::size_t>(vertex);
      const mask_t improved = candidates & ~visited[index];
      next_frontier[index] = improved;
      if (improved == 0)
        continue;

      const auto count =
          static_cast<unsigned int>(__builtin_popcountll(improved));
      const auto old_count = static_cast<unsigned int>(reached_counts[index]);
      const auto new_count = old_count + count;
      visited[index] |= improved;
      distance_sums[index] += static_cast<std::uint32_t>(level + 1) * count;
      reached_counts[index] = static_cast<unsigned char>(new_count);
      if (old_count == 0) {
        minimum_distances[index] =
            static_cast<unsigned char>(std::min(level + 1, 254));
      }
      if (old_count < median_target && new_count >= median_target) {
        median_distances[index] =
            static_cast<unsigned char>(std::min(level + 1, 254));
      }
      if (old_count < p75_target && new_count >= p75_target) {
        p75_distances[index] =
            static_cast<unsigned char>(std::min(level + 1, 254));
      }
      maximum_distances[index] =
          static_cast<unsigned char>(std::min(level + 1, 254));
      next_pairs += count;
      ++next_vertices;
    }
    if (next_pairs == 0)
      break;
    reached_pairs += next_pairs;
    ++level;
    frontier.swap(next_frontier);
    std::cout << "query_bfs_level=" << level
              << " active_vertices=" << next_vertices
              << " active_pairs=" << next_pairs << '\n';
  }

  const std::uint32_t unreachable_penalty =
      static_cast<std::uint32_t>(level + 1);
  std::vector<std::uint32_t> scores(vertex_count, 0);
  std::vector<std::uint32_t> ordering_keys(vertex_count, 0);
  std::uint32_t maximum_score = 0;
#pragma omp parallel for reduction(max : maximum_score) schedule(static)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const auto index = static_cast<std::size_t>(vertex);
    const auto missing =
        static_cast<std::uint32_t>(sources.size() - reached_counts[index]);
    if (score_mode == "min") {
      scores[index] = minimum_distances[index] == 255
                          ? unreachable_penalty
                          : minimum_distances[index];
    } else if (score_mode == "median") {
      scores[index] = median_distances[index] == 255 ? unreachable_penalty
                                                     : median_distances[index];
    } else if (score_mode == "p75") {
      scores[index] = p75_distances[index] == 255 ? unreachable_penalty
                                                  : p75_distances[index];
    } else if (score_mode == "max") {
      scores[index] =
          missing == 0 ? maximum_distances[index] : unreachable_penalty;
    } else {
      scores[index] = distance_sums[index] + missing * unreachable_penalty;
    }
    scores[index] /= static_cast<std::uint32_t>(score_bucket);
    maximum_score = std::max(maximum_score, scores[index]);
  }

  const std::uint32_t score_stride = maximum_score + 1;
  std::uint32_t maximum_key = 0;
#pragma omp parallel for reduction(max : maximum_key) schedule(static)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const auto index = static_cast<std::size_t>(vertex);
    const std::uint32_t minimum = minimum_distances[index] == 255
                                      ? unreachable_penalty
                                      : minimum_distances[index];
    ordering_keys[index] = primary_min_distance
                               ? minimum * score_stride + scores[index]
                               : scores[index];
    maximum_key = std::max(maximum_key, ordering_keys[index]);
  }

  std::vector<unsigned char> is_source(vertex_count, 0);
  std::vector<int> order;
  order.reserve(vertex_count);
  for (int source : sources) {
    const auto index = static_cast<std::size_t>(source);
    if (is_source[index] == 0) {
      is_source[index] = 1;
      order.push_back(source);
    }
  }

  std::vector<std::size_t> counts(static_cast<std::size_t>(maximum_key) + 1, 0);
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    if (is_source[static_cast<std::size_t>(vertex)] == 0) {
      ++counts[ordering_keys[static_cast<std::size_t>(vertex)]];
    }
  }
  std::vector<std::size_t> positions(counts.size(), order.size());
  std::size_t position = order.size();
  if (!reverse_score) {
    for (std::size_t score = 0; score < counts.size(); ++score) {
      positions[score] = position;
      position += counts[score];
    }
  } else {
    for (std::size_t reverse = counts.size(); reverse > 0; --reverse) {
      const std::size_t score = reverse - 1;
      positions[score] = position;
      position += counts[score];
    }
  }
  order.resize(vertex_count);

  auto place_vertex = [&](int vertex) {
    const auto index = static_cast<std::size_t>(vertex);
    if (is_source[index] == 0) {
      order[positions[ordering_keys[index]]++] = vertex;
    }
  };
  if (secondary_order != nullptr) {
    for (int vertex : *secondary_order)
      place_vertex(vertex);
  } else {
    for (int vertex = 0; vertex < graph.vertices; ++vertex) {
      place_vertex(vertex);
    }
  }

  std::cout << "query_bfs_levels=" << level
            << " reached_pairs=" << reached_pairs
            << " total_pairs=" << vertex_count * sources.size()
            << " maximum_score=" << maximum_score
            << " maximum_key=" << maximum_key << " score_mode=" << score_mode
            << " score_bucket=" << score_bucket
            << " primary_min_distance=" << (primary_min_distance ? 1 : 0)
            << " reverse_score=" << (reverse_score ? 1 : 0)
            << " secondary_order=" << (secondary_order != nullptr ? 1 : 0)
            << '\n';
  return order;
}

std::vector<int> stable_radix_order(const std::vector<std::uint64_t> &keys,
                                    const std::vector<int> *secondary_order,
                                    int key_bits) {
  std::vector<int> order(keys.size());
  if (secondary_order != nullptr) {
    order = *secondary_order;
  } else {
    std::iota(order.begin(), order.end(), 0);
  }
  std::vector<int> scratch(order.size());
  const int passes = (key_bits + 7) / 8;
  for (int pass = 0; pass < passes; ++pass) {
    const int shift = pass * 8;
    std::array<std::size_t, 256> counts{};
    for (int vertex : order) {
      ++counts[(keys[static_cast<std::size_t>(vertex)] >> shift) & 0xffU];
    }
    std::array<std::size_t, 256> positions{};
    for (std::size_t bucket = 1; bucket < positions.size(); ++bucket) {
      positions[bucket] = positions[bucket - 1] + counts[bucket - 1];
    }
    for (int vertex : order) {
      const auto bucket =
          (keys[static_cast<std::size_t>(vertex)] >> shift) & 0xffU;
      scratch[positions[bucket]++] = vertex;
    }
    order.swap(scratch);
  }
  return order;
}

std::vector<int>
query_coactivation_order(const csr_t &graph, const std::vector<int> &sources,
                         const std::vector<int> *secondary_order,
                         int profile_group_size, int phase_bits,
                         const std::string &score_mode, int score_bucket,
                         bool primary_min_distance) {
  using mask_t = std::uint64_t;
  const auto vertex_count = static_cast<std::size_t>(graph.vertices);
  const int dimensions =
      (static_cast<int>(sources.size()) + profile_group_size - 1) /
      profile_group_size;
  const int feature_dimensions = dimensions * (primary_min_distance ? 2 : 1);
  const auto maximum_phase = (std::uint32_t{1} << phase_bits) - 1;

  std::vector<std::uint64_t> keys(vertex_count, 0);
  std::vector<mask_t> frontier(vertex_count, 0);
  std::vector<mask_t> next_frontier(vertex_count, 0);
  std::vector<mask_t> visited(vertex_count, 0);
  std::vector<std::uint32_t> distance_sums(vertex_count, 0);
  std::vector<unsigned char> reached_counts(vertex_count, 0);
  std::vector<unsigned char> minimum_distances(vertex_count, 255);
  std::vector<unsigned char> median_distances(vertex_count, 255);
  std::vector<unsigned char> p75_distances(vertex_count, 255);
  std::vector<unsigned char> maximum_distances(vertex_count, 0);

  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const int first = dimension * profile_group_size;
    const int last =
        std::min(first + profile_group_size, static_cast<int>(sources.size()));
    const int group_queries = last - first;
    const auto median_target =
        static_cast<unsigned int>((group_queries + 1) / 2);
    const auto p75_target =
        static_cast<unsigned int>((3 * group_queries + 3) / 4);

    std::fill(frontier.begin(), frontier.end(), 0);
    std::fill(next_frontier.begin(), next_frontier.end(), 0);
    std::fill(visited.begin(), visited.end(), 0);
    std::fill(distance_sums.begin(), distance_sums.end(), 0);
    std::fill(reached_counts.begin(), reached_counts.end(), 0);
    std::fill(minimum_distances.begin(), minimum_distances.end(), 255);
    std::fill(median_distances.begin(), median_distances.end(), 255);
    std::fill(p75_distances.begin(), p75_distances.end(), 255);
    std::fill(maximum_distances.begin(), maximum_distances.end(), 0);

    for (int query = 0; query < group_queries; ++query) {
      const mask_t bit = mask_t{1} << query;
      const auto source = static_cast<std::size_t>(
          sources[static_cast<std::size_t>(first + query)]);
      frontier[source] |= bit;
      visited[source] |= bit;
      const auto old_count =
          static_cast<unsigned int>(reached_counts[source]++);
      const auto new_count = old_count + 1;
      minimum_distances[source] = 0;
      if (old_count < median_target && new_count >= median_target)
        median_distances[source] = 0;
      if (old_count < p75_target && new_count >= p75_target)
        p75_distances[source] = 0;
    }

    int level = 0;
    std::uint64_t reached_pairs = static_cast<std::uint64_t>(group_queries);
    while (true) {
      std::uint64_t next_pairs = 0;
#pragma omp parallel for reduction(+ : next_pairs) schedule(dynamic, 1024)
      for (int vertex = 0; vertex < graph.vertices; ++vertex) {
        mask_t candidates = 0;
        for (int edge = graph.row_offsets[vertex];
             edge < graph.row_offsets[vertex + 1]; ++edge) {
          candidates |= frontier[static_cast<std::size_t>(
              graph.columns[static_cast<std::size_t>(edge)])];
        }
        const auto index = static_cast<std::size_t>(vertex);
        const mask_t improved = candidates & ~visited[index];
        next_frontier[index] = improved;
        if (improved == 0)
          continue;

        const auto count =
            static_cast<unsigned int>(__builtin_popcountll(improved));
        const auto old_count = static_cast<unsigned int>(reached_counts[index]);
        const auto new_count = old_count + count;
        visited[index] |= improved;
        distance_sums[index] += static_cast<std::uint32_t>(level + 1) * count;
        reached_counts[index] = static_cast<unsigned char>(new_count);
        if (old_count == 0) {
          minimum_distances[index] =
              static_cast<unsigned char>(std::min(level + 1, 254));
        }
        if (old_count < median_target && new_count >= median_target) {
          median_distances[index] =
              static_cast<unsigned char>(std::min(level + 1, 254));
        }
        if (old_count < p75_target && new_count >= p75_target) {
          p75_distances[index] =
              static_cast<unsigned char>(std::min(level + 1, 254));
        }
        maximum_distances[index] =
            static_cast<unsigned char>(std::min(level + 1, 254));
        next_pairs += count;
      }
      if (next_pairs == 0)
        break;
      reached_pairs += next_pairs;
      ++level;
      frontier.swap(next_frontier);
    }

    const auto unreachable_penalty =
        static_cast<std::uint32_t>(std::min(level + 1, 254));
#pragma omp parallel for schedule(static)
    for (int vertex = 0; vertex < graph.vertices; ++vertex) {
      const auto index = static_cast<std::size_t>(vertex);
      const auto missing =
          static_cast<std::uint32_t>(group_queries - reached_counts[index]);
      std::uint32_t phase = 0;
      if (score_mode == "min") {
        phase = minimum_distances[index] == 255 ? unreachable_penalty
                                                : minimum_distances[index];
      } else if (score_mode == "median") {
        phase = median_distances[index] == 255 ? unreachable_penalty
                                               : median_distances[index];
      } else if (score_mode == "p75") {
        phase = p75_distances[index] == 255 ? unreachable_penalty
                                            : p75_distances[index];
      } else if (score_mode == "max") {
        phase = missing == 0 ? maximum_distances[index] : unreachable_penalty;
      } else {
        phase = (distance_sums[index] + missing * unreachable_penalty) /
                static_cast<std::uint32_t>(group_queries);
      }
      phase = std::min(phase / static_cast<std::uint32_t>(score_bucket),
                       maximum_phase);
      const auto minimum_phase =
          std::min((minimum_distances[index] == 255 ? unreachable_penalty
                                                    : minimum_distances[index]),
                   maximum_phase);
      for (int bit = 0; bit < phase_bits; ++bit) {
        const int target_bit = bit * feature_dimensions + dimension;
        keys[index] |= ((static_cast<std::uint64_t>(phase) >> bit) & 1U)
                       << target_bit;
        if (primary_min_distance) {
          const int minimum_target_bit =
              bit * feature_dimensions + dimensions + dimension;
          keys[index] |=
              ((static_cast<std::uint64_t>(minimum_phase) >> bit) & 1U)
              << minimum_target_bit;
        }
      }
    }
    std::cout << "coactivation_dimension=" << dimension
              << " queries=" << group_queries << " levels=" << level
              << " reached_pairs=" << reached_pairs << '\n';
  }

  auto order = stable_radix_order(keys, secondary_order,
                                  feature_dimensions * phase_bits);
  std::vector<unsigned char> is_source(vertex_count, 0);
  std::vector<int> result;
  result.reserve(vertex_count);
  for (int source : sources) {
    const auto index = static_cast<std::size_t>(source);
    if (is_source[index] == 0) {
      is_source[index] = 1;
      result.push_back(source);
    }
  }
  std::uint64_t distinct_keys = 0;
  std::uint64_t previous_key = 0;
  bool first_key = true;
  for (int vertex : order) {
    const auto index = static_cast<std::size_t>(vertex);
    if (is_source[index] == 0)
      result.push_back(vertex);
    const auto key = keys[index];
    if (first_key || key != previous_key) {
      ++distinct_keys;
      previous_key = key;
      first_key = false;
    }
  }
  std::cout << "coactivation_dimensions=" << dimensions
            << " profile_group_size=" << profile_group_size
            << " phase_bits=" << phase_bits << " score_mode=" << score_mode
            << " score_bucket=" << score_bucket
            << " primary_min_distance=" << (primary_min_distance ? 1 : 0)
            << " distinct_keys=" << distinct_keys
            << " secondary_order=" << (secondary_order != nullptr ? 1 : 0)
            << '\n';
  return result;
}

std::vector<int> source_owner_order(const csr_t &graph,
                                    const std::vector<int> &sources) {
  std::vector<int> owner(static_cast<std::size_t>(graph.vertices), -1);
  std::vector<int> queue;
  queue.reserve(static_cast<std::size_t>(graph.vertices));
  for (int source_id = 0; source_id < static_cast<int>(sources.size());
       ++source_id) {
    const int source = sources[static_cast<std::size_t>(source_id)];
    if (owner[static_cast<std::size_t>(source)] == -1) {
      owner[static_cast<std::size_t>(source)] = source_id;
      queue.push_back(source);
    }
  }
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int vertex = queue[head];
    const int vertex_owner = owner[static_cast<std::size_t>(vertex)];
    for (int edge = graph.row_offsets[vertex];
         edge < graph.row_offsets[vertex + 1]; ++edge) {
      const int neighbor = graph.columns[static_cast<std::size_t>(edge)];
      if (owner[static_cast<std::size_t>(neighbor)] == -1) {
        owner[static_cast<std::size_t>(neighbor)] = vertex_owner;
        queue.push_back(neighbor);
      }
    }
  }
  std::vector<std::size_t> counts(sources.size() + 1, 0);
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const int vertex_owner = owner[static_cast<std::size_t>(vertex)];
    ++counts[vertex_owner < 0 ? sources.size()
                              : static_cast<std::size_t>(vertex_owner)];
  }
  std::vector<std::size_t> positions(counts.size(), 0);
  for (std::size_t owner_id = 1; owner_id < counts.size(); ++owner_id) {
    positions[owner_id] = positions[owner_id - 1] + counts[owner_id - 1];
  }
  std::vector<int> order(static_cast<std::size_t>(graph.vertices));
  for (int vertex : queue) {
    const auto owner_id =
        static_cast<std::size_t>(owner[static_cast<std::size_t>(vertex)]);
    order[positions[owner_id]++] = vertex;
  }
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    if (owner[static_cast<std::size_t>(vertex)] < 0) {
      order[positions.back()++] = vertex;
    }
  }
  std::cout << "owner_groups=" << sources.size()
            << " owner_reached=" << queue.size() << '\n';
  return order;
}

class radix_heap_t {
public:
  using key_t = std::uint32_t;
  using item_t = std::pair<key_t, int>;

  bool empty() const { return size_ == 0; }

  void push(key_t key, int value) {
    if (key < last_) {
      throw std::logic_error("radix heap received a decreasing key");
    }
    buckets_[bucket_index(key, last_)].emplace_back(key, value);
    ++size_;
  }

  item_t pop() {
    if (buckets_[0].empty()) {
      std::size_t bucket = 1;
      while (bucket < buckets_.size() && buckets_[bucket].empty())
        ++bucket;
      if (bucket == buckets_.size()) {
        throw std::logic_error("pop from empty radix heap");
      }
      key_t next_last = std::numeric_limits<key_t>::max();
      for (const auto &item : buckets_[bucket]) {
        next_last = std::min(next_last, item.first);
      }
      last_ = next_last;
      auto items = std::move(buckets_[bucket]);
      buckets_[bucket].clear();
      for (const auto &item : items) {
        buckets_[bucket_index(item.first, last_)].push_back(item);
      }
    }
    auto result = buckets_[0].back();
    buckets_[0].pop_back();
    --size_;
    return result;
  }

private:
  static std::size_t bucket_index(key_t key, key_t last) {
    const key_t difference = key ^ last;
    return difference == 0
               ? 0
               : static_cast<std::size_t>(32 - __builtin_clz(difference));
  }

  std::array<std::vector<item_t>, 33> buckets_;
  std::size_t size_ = 0;
  key_t last_ = 0;
};

std::vector<int> sssp_order(const csr_t &graph, const std::vector<int> &sources,
                            int distance_bucket,
                            const std::vector<int> *secondary_rank,
                            std::vector<std::size_t> *segment_offsets) {
  using distance_t = std::uint32_t;
  constexpr distance_t infinity = std::numeric_limits<distance_t>::max();
  std::vector<distance_t> distances(static_cast<std::size_t>(graph.vertices),
                                    infinity);
  std::vector<unsigned char> settled(static_cast<std::size_t>(graph.vertices),
                                     0);
  radix_heap_t queue;
  for (auto source = sources.rbegin(); source != sources.rend(); ++source) {
    if (distances[static_cast<std::size_t>(*source)] == infinity) {
      distances[static_cast<std::size_t>(*source)] = 0;
      queue.push(0, *source);
    }
  }

  std::vector<int> order;
  order.reserve(static_cast<std::size_t>(graph.vertices));
  distance_t maximum_distance = 0;
  while (!queue.empty()) {
    const auto [distance, vertex] = queue.pop();
    if (settled[static_cast<std::size_t>(vertex)] != 0 ||
        distance != distances[static_cast<std::size_t>(vertex)]) {
      continue;
    }
    settled[static_cast<std::size_t>(vertex)] = 1;
    order.push_back(vertex);
    maximum_distance = std::max(maximum_distance, distance);
    for (int edge = graph.row_offsets[vertex];
         edge < graph.row_offsets[vertex + 1]; ++edge) {
      const int neighbor = graph.columns[static_cast<std::size_t>(edge)];
      const int weight = graph.weights.empty()
                             ? 1
                             : graph.weights[static_cast<std::size_t>(edge)];
      if (weight < 0) {
        throw std::runtime_error("source-sssp requires nonnegative weights");
      }
      const auto candidate64 = static_cast<std::uint64_t>(distance) +
                               static_cast<std::uint64_t>(weight);
      const auto candidate = candidate64 >= infinity
                                 ? infinity
                                 : static_cast<distance_t>(candidate64);
      if (candidate < distances[static_cast<std::size_t>(neighbor)]) {
        distances[static_cast<std::size_t>(neighbor)] = candidate;
        queue.push(candidate, neighbor);
      }
    }
  }
  const auto reached = order.size();
  if (secondary_rank != nullptr) {
    std::size_t begin = 0;
    while (begin < reached) {
      const auto bucket = distances[static_cast<std::size_t>(order[begin])] /
                          static_cast<distance_t>(distance_bucket);
      std::size_t end = begin + 1;
      while (end < reached &&
             distances[static_cast<std::size_t>(order[end])] /
                     static_cast<distance_t>(distance_bucket) ==
                 bucket) {
        ++end;
      }
      std::sort(order.begin() + static_cast<std::ptrdiff_t>(begin),
                order.begin() + static_cast<std::ptrdiff_t>(end),
                [&](int left, int right) {
                  return (*secondary_rank)[static_cast<std::size_t>(left)] <
                         (*secondary_rank)[static_cast<std::size_t>(right)];
                });
      begin = end;
    }
  }
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    if (settled[static_cast<std::size_t>(vertex)] == 0)
      order.push_back(vertex);
  }
  if (segment_offsets != nullptr) {
    segment_offsets->clear();
    segment_offsets->push_back(0);
    if (reached != 0) {
      auto previous_bucket =
          distances[static_cast<std::size_t>(order[0])] /
          static_cast<distance_t>(distance_bucket);
      for (std::size_t index = 1; index < reached; ++index) {
        const auto bucket = distances[static_cast<std::size_t>(order[index])] /
                            static_cast<distance_t>(distance_bucket);
        if (bucket != previous_bucket) {
          segment_offsets->push_back(index);
          previous_bucket = bucket;
        }
      }
    }
    if (segment_offsets->back() != reached)
      segment_offsets->push_back(reached);
    if (segment_offsets->back() != order.size())
      segment_offsets->push_back(order.size());
  }
  std::cout << "sssp_reached=" << reached
            << " sssp_max_distance=" << maximum_distance
            << " distance_bucket=" << distance_bucket
            << " seeds=" << sources.size() << '\n';
  return order;
}

std::vector<int> hub_cluster_order(const csr_t &graph, int requested_hubs) {
  const int hub_count = std::min(requested_hubs, graph.vertices);
  auto degree_sorted = degree_order(graph, true);
  degree_sorted.resize(static_cast<std::size_t>(hub_count));
  std::vector<int> hub_rank(static_cast<std::size_t>(graph.vertices), -1);
  for (int rank = 0; rank < hub_count; ++rank) {
    hub_rank[static_cast<std::size_t>(degree_sorted[rank])] = rank;
  }
  std::vector<int> group(static_cast<std::size_t>(graph.vertices), hub_count);
#pragma omp parallel for schedule(dynamic, 1024)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    int best = hub_rank[static_cast<std::size_t>(vertex)];
    if (best < 0)
      best = hub_count;
    for (int edge = graph.row_offsets[vertex];
         edge < graph.row_offsets[vertex + 1]; ++edge) {
      const int rank = hub_rank[static_cast<std::size_t>(graph.columns[edge])];
      if (rank >= 0 && rank < best)
        best = rank;
    }
    group[static_cast<std::size_t>(vertex)] = best;
  }
  std::vector<std::size_t> counts(static_cast<std::size_t>(hub_count) + 1, 0);
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    if (hub_rank[static_cast<std::size_t>(vertex)] < 0) {
      ++counts[static_cast<std::size_t>(
          group[static_cast<std::size_t>(vertex)])];
    }
  }
  std::vector<std::size_t> positions(counts.size(), 0);
  std::size_t position = static_cast<std::size_t>(hub_count);
  for (std::size_t group_id = 0; group_id < counts.size(); ++group_id) {
    positions[group_id] = position;
    position += counts[group_id];
  }
  std::vector<int> order(static_cast<std::size_t>(graph.vertices));
  std::copy(degree_sorted.begin(), degree_sorted.end(), order.begin());
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    if (hub_rank[static_cast<std::size_t>(vertex)] < 0) {
      const auto group_id =
          static_cast<std::size_t>(group[static_cast<std::size_t>(vertex)]);
      order[positions[group_id]++] = vertex;
    }
  }
  return order;
}

std::vector<int> hub_order(const csr_t &graph, int requested_threshold,
                           bool sort_hubs) {
  const int threshold =
      requested_threshold > 0
          ? requested_threshold
          : std::max(1, static_cast<int>(
                            std::ceil(static_cast<double>(graph.edges) /
                                      static_cast<double>(graph.vertices))));
  std::vector<int> hubs;
  std::vector<int> non_hubs;
  hubs.reserve(static_cast<std::size_t>(graph.vertices) / 16);
  non_hubs.reserve(static_cast<std::size_t>(graph.vertices));
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    const int degree =
        graph.row_offsets[vertex + 1] - graph.row_offsets[vertex];
    (degree >= threshold ? hubs : non_hubs).push_back(vertex);
  }
  if (sort_hubs) {
    std::sort(hubs.begin(), hubs.end(), [&](int left, int right) {
      const int left_degree =
          graph.row_offsets[left + 1] - graph.row_offsets[left];
      const int right_degree =
          graph.row_offsets[right + 1] - graph.row_offsets[right];
      return left_degree != right_degree ? left_degree > right_degree
                                         : left < right;
    });
  }
  std::vector<int> order;
  order.reserve(static_cast<std::size_t>(graph.vertices));
  order.insert(order.end(), hubs.begin(), hubs.end());
  order.insert(order.end(), non_hubs.begin(), non_hubs.end());
  std::cout << "hub_threshold=" << threshold << " hubs=" << hubs.size()
            << " sort_hubs=" << (sort_hubs ? 1 : 0) << '\n';
  return order;
}

std::uint64_t mix_hash(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

std::vector<int> minhash_order(const csr_t &graph, unsigned int seed) {
  struct signature_t {
    std::uint64_t first;
    std::uint64_t second;
    int vertex;
  };
  std::vector<signature_t> signatures(static_cast<std::size_t>(graph.vertices));
  const std::uint64_t seed0 =
      mix_hash(static_cast<std::uint64_t>(seed) ^ 0x243f6a8885a308d3ULL);
  const std::uint64_t seed1 =
      mix_hash(static_cast<std::uint64_t>(seed) ^ 0x13198a2e03707344ULL);
#pragma omp parallel for schedule(dynamic, 1024)
  for (int vertex = 0; vertex < graph.vertices; ++vertex) {
    std::uint64_t first = mix_hash(static_cast<std::uint64_t>(vertex) ^ seed0);
    std::uint64_t second = mix_hash(static_cast<std::uint64_t>(vertex) ^ seed1);
    for (int edge = graph.row_offsets[vertex];
         edge < graph.row_offsets[vertex + 1]; ++edge) {
      const auto neighbor = static_cast<std::uint64_t>(
          graph.columns[static_cast<std::size_t>(edge)]);
      first = std::min(first, mix_hash(neighbor ^ seed0));
      second = std::min(second, mix_hash(neighbor ^ seed1));
    }
    signatures[static_cast<std::size_t>(vertex)] = {first, second, vertex};
  }
  std::sort(signatures.begin(), signatures.end(),
            [](const signature_t &left, const signature_t &right) {
              if (left.first != right.first)
                return left.first < right.first;
              if (left.second != right.second)
                return left.second < right.second;
              return left.vertex < right.vertex;
            });
  std::vector<int> order(static_cast<std::size_t>(graph.vertices));
#pragma omp parallel for schedule(static)
  for (int index = 0; index < graph.vertices; ++index) {
    order[static_cast<std::size_t>(index)] =
        signatures[static_cast<std::size_t>(index)].vertex;
  }
  return order;
}

int strategy_tag(const std::string &strategy) {
  if (strategy == "degree-desc")
    return 10;
  if (strategy == "degree-asc")
    return 11;
  if (strategy == "bucket-desc")
    return 12;
  if (strategy == "bucket-asc")
    return 13;
  if (strategy == "bfs")
    return 20;
  if (strategy == "rcm")
    return 21;
  if (strategy == "source-bfs")
    return 30;
  if (strategy == "source-bfs-reverse")
    return 31;
  if (strategy == "source-sssp")
    return 32;
  if (strategy == "source-owner")
    return 33;
  if (strategy == "query-bfs-borda")
    return 34;
  if (strategy == "query-bfs-borda-reverse")
    return 35;
  if (strategy == "query-bfs-profile")
    return 36;
  if (strategy == "query-coactivation")
    return 37;
  if (strategy == "hub-cluster")
    return 40;
  if (strategy == "hub-sort")
    return 41;
  if (strategy == "hub-group")
    return 42;
  if (strategy == "minhash")
    return 43;
  if (strategy == "random")
    return 50;
  return 0;
}

std::vector<int> invert_and_validate(const std::vector<int> &new_to_old);

void sort_degree_in_local_windows(std::vector<int> &order, const csr_t &graph,
                                  int window_size,
                                  const std::vector<std::size_t> *segments) {
  if (window_size <= 1)
    return;
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::pair<std::size_t, std::size_t>> windows;
  auto append_windows = [&](std::size_t segment_begin,
                            std::size_t segment_end) {
    for (std::size_t begin = segment_begin; begin < segment_end;
         begin += static_cast<std::size_t>(window_size)) {
      windows.emplace_back(
          begin, std::min(begin + static_cast<std::size_t>(window_size),
                          segment_end));
    }
  };
  if (segments != nullptr && segments->size() >= 2) {
    for (std::size_t segment = 0; segment + 1 < segments->size(); ++segment) {
      append_windows((*segments)[segment], (*segments)[segment + 1]);
    }
  } else {
    append_windows(0, order.size());
  }
#pragma omp parallel for schedule(dynamic, 64)
  for (std::int64_t window = 0;
       window < static_cast<std::int64_t>(windows.size()); ++window) {
    const auto begin = windows[static_cast<std::size_t>(window)].first;
    const auto end = windows[static_cast<std::size_t>(window)].second;
    std::stable_sort(
        order.begin() + begin, order.begin() + end, [&](int left, int right) {
          const int left_degree =
              graph.row_offsets[left + 1] - graph.row_offsets[left];
          const int right_degree =
              graph.row_offsets[right + 1] - graph.row_offsets[right];
          return left_degree < right_degree;
        });
  }
  std::cout << "local_degree_sort_ms=" << elapsed_ms(start)
            << " local_degree_window=" << window_size
            << " segment_aware="
            << (segments != nullptr && segments->size() >= 2 ? 1 : 0) << '\n';
}

void arrange_sources(std::vector<int> &sources, const csr_t &graph,
                     const options_t &options) {
  if (options.source_order == "input")
    return;
  if (options.source_order == "shuffle") {
    std::mt19937 random(options.seed);
    std::shuffle(sources.begin(), sources.end(), random);
    return;
  }
  std::sort(sources.begin(), sources.end(), [&](int left, int right) {
    const int left_degree =
        graph.row_offsets[left + 1] - graph.row_offsets[left];
    const int right_degree =
        graph.row_offsets[right + 1] - graph.row_offsets[right];
    if (left_degree != right_degree) {
      return options.source_order == "degree-desc" ? left_degree > right_degree
                                                   : left_degree < right_degree;
    }
    return left < right;
  });
}

std::vector<int> make_order(const csr_t &graph, const options_t &options,
                            int *tag,
                            std::vector<std::size_t> *segment_offsets) {
  auto start = std::chrono::steady_clock::now();
  std::vector<int> order;
  std::vector<int> secondary_order;
  std::vector<int> secondary_rank;
  if (options.within_level == "mapping") {
    const auto secondary =
        load_mapping(options.secondary_mapping, graph.vertices);
    secondary_order = secondary.new_to_old;
    if (options.strategy == "source-sssp")
      secondary_rank = invert_and_validate(secondary_order);
  }
  *tag = strategy_tag(options.strategy);
  if (options.strategy == "mapping") {
    auto mapping = load_mapping(options.mapping_input, graph.vertices);
    *tag = mapping.strategy;
    order = std::move(mapping.new_to_old);
  } else if (options.strategy == "degree-desc") {
    order = degree_order(graph, true);
  } else if (options.strategy == "degree-asc") {
    order = degree_order(graph, false);
  } else if (options.strategy == "bucket-desc") {
    order = bucket_order(graph, true);
  } else if (options.strategy == "bucket-asc") {
    order = bucket_order(graph, false);
  } else if (options.strategy == "bfs" || options.strategy == "rcm") {
    int root = 0;
    for (int vertex = 1; vertex < graph.vertices; ++vertex) {
      if (graph.row_offsets[vertex + 1] - graph.row_offsets[vertex] >
          graph.row_offsets[root + 1] - graph.row_offsets[root]) {
        root = vertex;
      }
    }
    order = bfs_order(graph, {root}, options.within_level,
                      options.strategy == "rcm",
                      secondary_order.empty() ? nullptr : &secondary_order,
                      options.level_bucket);
  } else if (options.strategy == "source-bfs" ||
             options.strategy == "source-bfs-reverse") {
    auto sources =
        load_sources(options.sources_path, options.query_count, graph.vertices);
    arrange_sources(sources, graph, options);
    order = bfs_order(graph, sources, options.within_level,
                      options.strategy == "source-bfs-reverse",
                      secondary_order.empty() ? nullptr : &secondary_order,
                      options.level_bucket,
                      options.segment_offsets_output.empty()
                          ? nullptr
                          : segment_offsets);
  } else if (options.strategy == "source-sssp") {
    auto sources =
        load_sources(options.sources_path, options.query_count, graph.vertices);
    arrange_sources(sources, graph, options);
    order = sssp_order(graph, sources, options.distance_bucket,
                       secondary_rank.empty() ? nullptr : &secondary_rank,
                       options.segment_offsets_output.empty()
                           ? nullptr
                           : segment_offsets);
  } else if (options.strategy == "query-bfs-borda" ||
             options.strategy == "query-bfs-borda-reverse" ||
             options.strategy == "query-bfs-profile") {
    auto sources =
        load_sources(options.sources_path, options.query_count, graph.vertices);
    arrange_sources(sources, graph, options);
    const std::vector<int> *secondary_order = nullptr;
    mapping_file_t secondary;
    if (!options.secondary_mapping.empty()) {
      secondary = load_mapping(options.secondary_mapping, graph.vertices);
      secondary_order = &secondary.new_to_old;
    }
    order = query_bfs_borda_order(
        graph, sources, secondary_order,
        options.strategy == "query-bfs-borda-reverse", options.query_score,
        options.primary_min_distance, options.query_score_bucket);
  } else if (options.strategy == "query-coactivation") {
    auto sources =
        load_sources(options.sources_path, options.query_count, graph.vertices);
    arrange_sources(sources, graph, options);
    const std::vector<int> *secondary_order = nullptr;
    mapping_file_t secondary;
    if (!options.secondary_mapping.empty()) {
      secondary = load_mapping(options.secondary_mapping, graph.vertices);
      secondary_order = &secondary.new_to_old;
    }
    order = query_coactivation_order(
        graph, sources, secondary_order, options.profile_group_size,
        options.phase_bits, options.query_score, options.query_score_bucket,
        options.primary_min_distance);
  } else if (options.strategy == "source-owner") {
    auto sources =
        load_sources(options.sources_path, options.query_count, graph.vertices);
    arrange_sources(sources, graph, options);
    order = source_owner_order(graph, sources);
  } else if (options.strategy == "hub-cluster") {
    order = hub_cluster_order(graph, options.hub_count);
  } else if (options.strategy == "hub-sort") {
    order = hub_order(graph, options.hub_threshold, true);
  } else if (options.strategy == "hub-group") {
    order = hub_order(graph, options.hub_threshold, false);
  } else if (options.strategy == "minhash") {
    order = minhash_order(graph, options.seed);
  } else if (options.strategy == "random") {
    order.resize(static_cast<std::size_t>(graph.vertices));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(options.seed);
    std::shuffle(order.begin(), order.end(), random);
  }
  std::cout << "order_ms=" << elapsed_ms(start)
            << " strategy=" << options.strategy << " strategy_tag=" << *tag
            << '\n';
  return order;
}

std::vector<int> invert_and_validate(const std::vector<int> &new_to_old) {
  std::vector<int> old_to_new(new_to_old.size(), -1);
  for (std::size_t new_id = 0; new_id < new_to_old.size(); ++new_id) {
    const int old_id = new_to_old[new_id];
    if (old_id < 0 || old_id >= static_cast<int>(new_to_old.size()) ||
        old_to_new[static_cast<std::size_t>(old_id)] != -1) {
      throw std::runtime_error("generated order is not a permutation");
    }
    old_to_new[static_cast<std::size_t>(old_id)] = static_cast<int>(new_id);
  }
  return old_to_new;
}

void reorder_neighbors(csr_t &graph, const std::string &neighbor_order) {
  if (neighbor_order == "preserve")
    return;
  const auto start = std::chrono::steady_clock::now();
#pragma omp parallel
  {
    std::vector<std::pair<int, int>> row;
#pragma omp for schedule(dynamic, 1024)
    for (int vertex = 0; vertex < graph.vertices; ++vertex) {
      const int begin = graph.row_offsets[vertex];
      const int end = graph.row_offsets[vertex + 1];
      if (end - begin <= 1)
        continue;
      row.clear();
      row.reserve(static_cast<std::size_t>(end - begin));
      for (int edge = begin; edge < end; ++edge) {
        row.emplace_back(graph.columns[static_cast<std::size_t>(edge)],
                         graph.weights.empty()
                             ? 0
                             : graph.weights[static_cast<std::size_t>(edge)]);
      }
      std::sort(
          row.begin(), row.end(), [&](const auto &left, const auto &right) {
            if (neighbor_order == "id")
              return left < right;
            const int left_degree =
                graph.row_offsets[static_cast<std::size_t>(left.first) + 1] -
                graph.row_offsets[static_cast<std::size_t>(left.first)];
            const int right_degree =
                graph.row_offsets[static_cast<std::size_t>(right.first) + 1] -
                graph.row_offsets[static_cast<std::size_t>(right.first)];
            if (left_degree != right_degree) {
              return neighbor_order == "degree-desc"
                         ? left_degree > right_degree
                         : left_degree < right_degree;
            }
            return left < right;
          });
      for (int edge = begin; edge < end; ++edge) {
        const auto &item = row[static_cast<std::size_t>(edge - begin)];
        graph.columns[static_cast<std::size_t>(edge)] = item.first;
        if (!graph.weights.empty()) {
          graph.weights[static_cast<std::size_t>(edge)] = item.second;
        }
      }
    }
  }
  std::cout << "neighbor_sort_ms=" << elapsed_ms(start)
            << " neighbor_order=" << neighbor_order << '\n';
}

csr_t apply_mapping(const csr_t &input, const std::vector<int> &new_to_old,
                    const std::vector<int> &old_to_new,
                    const std::string &neighbor_order) {
  auto start = std::chrono::steady_clock::now();
  csr_t output;
  output.vertices = input.vertices;
  output.edges = input.edges;
  output.row_offsets.resize(input.row_offsets.size());
  output.columns.resize(input.columns.size());
  if (!input.weights.empty())
    output.weights.resize(input.weights.size());

  output.row_offsets[0] = 0;
  for (int new_id = 0; new_id < input.vertices; ++new_id) {
    const int old_id = new_to_old[static_cast<std::size_t>(new_id)];
    const int degree =
        input.row_offsets[old_id + 1] - input.row_offsets[old_id];
    output.row_offsets[static_cast<std::size_t>(new_id) + 1] =
        output.row_offsets[static_cast<std::size_t>(new_id)] + degree;
  }
  if (output.row_offsets.back() != input.edges) {
    throw std::runtime_error("reordered row offsets lost edges");
  }

#pragma omp parallel for schedule(dynamic, 1024)
  for (int new_id = 0; new_id < input.vertices; ++new_id) {
    const int old_id = new_to_old[static_cast<std::size_t>(new_id)];
    const int input_begin = input.row_offsets[old_id];
    const int input_end = input.row_offsets[old_id + 1];
    int output_edge = output.row_offsets[static_cast<std::size_t>(new_id)];
    for (int input_edge = input_begin; input_edge < input_end;
         ++input_edge, ++output_edge) {
      output.columns[static_cast<std::size_t>(output_edge)] =
          old_to_new[static_cast<std::size_t>(
              input.columns[static_cast<std::size_t>(input_edge)])];
      if (!input.weights.empty()) {
        output.weights[static_cast<std::size_t>(output_edge)] =
            input.weights[static_cast<std::size_t>(input_edge)];
      }
    }
  }
  reorder_neighbors(output, neighbor_order);
  std::cout << "apply_ms=" << elapsed_ms(start) << '\n';
  return output;
}

void print_locality_sample(const csr_t &graph) {
  const std::size_t target_samples = 10'000'000;
  const std::size_t stride =
      std::max<std::size_t>(1, graph.columns.size() / target_samples);
  long double span_sum = 0.0;
  std::uint64_t within_32 = 0;
  std::uint64_t within_256 = 0;
  std::uint64_t within_4096 = 0;
  std::uint64_t samples = 0;
  int vertex = 0;
  for (std::size_t edge = 0; edge < graph.columns.size(); edge += stride) {
    while (vertex + 1 < static_cast<int>(graph.row_offsets.size()) &&
           graph.row_offsets[static_cast<std::size_t>(vertex) + 1] <=
               static_cast<int>(edge)) {
      ++vertex;
    }
    const auto span = static_cast<std::uint64_t>(
        std::llabs(static_cast<long long>(vertex) - graph.columns[edge]));
    span_sum += static_cast<long double>(span);
    within_32 += span <= 32;
    within_256 += span <= 256;
    within_4096 += span <= 4096;
    ++samples;
  }
  std::cout << std::setprecision(9) << "locality_samples=" << samples
            << " avg_edge_span=" << static_cast<double>(span_sum / samples)
            << " within_32=" << static_cast<double>(within_32) / samples
            << " within_256=" << static_cast<double>(within_256) / samples
            << " within_4096=" << static_cast<double>(within_4096) / samples
            << '\n';
}

void write_csr(const fs::path &directory, const csr_t &graph) {
  fs::create_directories(directory);
  auto start = std::chrono::steady_clock::now();
  write_binary_vector(directory / "csr_vlist.bin", graph.row_offsets);
  write_binary_vector(directory / "csr_elist.bin", graph.columns);
  if (!graph.weights.empty()) {
    write_binary_vector(directory / "csr_weightlist.bin", graph.weights);
  }
  std::cout << "write_ms=" << elapsed_ms(start) << '\n';
}

void write_segment_offsets(const fs::path &path,
                           const std::vector<std::size_t> &offsets) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    throw std::runtime_error("cannot write segment offsets: " + path.string());
  for (const auto offset : offsets)
    output << offset << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse_options(argc, argv);
#ifdef _OPENMP
    if (options.threads > 0)
      omp_set_num_threads(options.threads);
    std::cout << "threads=" << omp_get_max_threads() << '\n';
#else
    std::cout << "threads=1\n";
#endif
    const auto total_start = std::chrono::steady_clock::now();
    auto input = load_csr(options.input_dir);
    int tag = 0;
    std::vector<std::size_t> segment_offsets;
    auto new_to_old = make_order(input, options, &tag, &segment_offsets);
    sort_degree_in_local_windows(new_to_old, input,
                                 options.local_degree_window,
                                 segment_offsets.empty() ? nullptr
                                                         : &segment_offsets);
    auto old_to_new = invert_and_validate(new_to_old);
    auto output =
        apply_mapping(input, new_to_old, old_to_new, options.neighbor_order);
    print_locality_sample(output);
    write_csr(options.output_dir, output);
    fs::create_directories(options.mapping_output.parent_path());
    write_mapping(options.mapping_output, tag, new_to_old);
    if (!options.segment_offsets_output.empty()) {
      write_segment_offsets(options.segment_offsets_output, segment_offsets);
      std::cout << "segment_offsets=" << options.segment_offsets_output
                << " segments=" << segment_offsets.size() - 1 << '\n';
    }
    std::cout << "mapping=" << options.mapping_output << '\n'
              << "output=" << options.output_dir << '\n'
              << "total_ms=" << elapsed_ms(total_start) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
