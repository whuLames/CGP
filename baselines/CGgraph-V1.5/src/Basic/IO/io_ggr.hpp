#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Type/data_type.hpp"

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

/* *********************************************************************************************************
 * GGR (Galois GR) 二进制图格式支持
 *
 * 文件布局:
 *   [Header 32B]  version(uint64) + sizeEdgeTy(uint64) + nvtxs(uint64) + nedges(uint64)
 *   [Data]        V×int64 row_start[1..V] + E×int32 edge_dst [+ 4B padding if weighted & E odd] + E×int32 adjwgt
 *
 * 注意: row_start 省略了 [0]（永远是 0），仅存储 [1..V]
 * *********************************************************************************************************/

namespace CPJ {

struct GGR_Header
{
    uint64_t version    = 0; // 必须为 1
    uint64_t sizeEdgeTy = 0; // 0=无权图, 非0=有权图
    uint64_t nvtxs      = 0; // 顶点数 V
    uint64_t nedges     = 0; // 边数 E
};

// 读取 GGR 文件的 32B header
inline GGR_Header readGGRHeader(const std::string& path)
{
    GGR_Header header;
    int fd = open(path.c_str(), O_RDONLY);
    assert_msg(fd >= 0, "Cannot open GGR file: %s", path.c_str());

    ssize_t n = read(fd, &header, sizeof(GGR_Header));
    close(fd);

    assert_msg(n == sizeof(GGR_Header), "Failed to read GGR header from: %s (read %zd/%zu bytes)",
               path.c_str(), n, sizeof(GGR_Header));
    return header;
}

// 判断文件是否为合法 GGR 格式
inline bool isGGRFile(const std::string& path)
{
    // 扩展名检查
    std::string ext = std::filesystem::path(path).extension().string();
    if (ext != ".ggr" && ext != ".gr")
        return false;

    // 文件大小至少 32B header
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || st.st_size < 32)
        return false;

    // 验证 version 字段
    GGR_Header header = readGGRHeader(path);
    return header.version == 1;
}

// 从 GGR 文件加载 CSR 数据到 CSR_Result_type
// needWeight: SSSP 等算法需要权重，BFS 不需要
inline void loadGGR(const std::string& path, CSR_Result_type& csrResult, bool needWeight)
{
    Msg_info("Loading GGR file: [%s]", path.c_str());

    // 1. 读取 header
    GGR_Header header = readGGRHeader(path);
    assert_msg(header.version == 1, "Unsupported GGR version: %lu (expected 1)", header.version);

    uint64_t V = header.nvtxs;
    uint64_t E = header.nedges;
    bool hasWeight = (header.sizeEdgeTy != 0);

    Msg_info("GGR Header: V=%lu, E=%lu, weighted=%s", V, E, hasWeight ? "yes" : "no");

    // 2. 类型溢出检查
    assert_msg(V < std::numeric_limits<count_type>::max(),
               "GGR: nvtxs(%lu) exceeds count_type range, set count_type to uint64_t", V);
    assert_msg(E < std::numeric_limits<countl_type>::max(),
               "GGR: nedges(%lu) exceeds countl_type range, set countl_type to uint64_t", E);
    assert_msg(V < std::numeric_limits<vertex_id_type>::max(),
               "GGR: nvtxs(%lu) exceeds vertex_id_type range", V);

    csrResult.vertexNum = static_cast<count_type>(V);
    csrResult.edgeNum = static_cast<countl_type>(E);

    // 3. 打开文件，跳到数据区（跳过 32B header）
    int fd = open(path.c_str(), O_RDONLY);
    assert_msg(fd >= 0, "Cannot open GGR file: %s", path.c_str());
    off_t ret = lseek(fd, 32, SEEK_SET);
    assert_msg(ret == 32, "Failed to seek in GGR file");

    // 4. 读取 offset: V 个 int64 → 转为 (V+1) 个 countl_type
    {
        int64_t* buf64 = new int64_t[V];
        ssize_t n = read(fd, buf64, sizeof(int64_t) * V);
        assert_msg(n == static_cast<ssize_t>(sizeof(int64_t) * V),
                   "Failed to read GGR row_start: expected %zu bytes, got %zd",
                   sizeof(int64_t) * V, n);

        csrResult.csr_offset = new countl_type[V + 1];
        csrResult.csr_offset[0] = 0; // GGR 省略了 [0]
        for (uint64_t i = 0; i < V; i++)
        {
            assert_msg(buf64[i] >= 0, "GGR: negative row_start[%lu] = %ld", i + 1, buf64[i]);
            assert_msg(static_cast<uint64_t>(buf64[i]) <= std::numeric_limits<countl_type>::max(),
                       "GGR: row_start[%lu] = %ld exceeds countl_type range", i + 1, buf64[i]);
            csrResult.csr_offset[i + 1] = static_cast<countl_type>(buf64[i]);
        }
        delete[] buf64;
    }
    Msg_info("GGR: csr_offset loaded (%zu bytes)", sizeof(countl_type) * (V + 1));

    // 5. 读取 dest: E 个 int32 → 转为 vertex_id_type (uint32)
    {
        csrResult.csr_dest = new vertex_id_type[E];
        ssize_t n = read(fd, csrResult.csr_dest, sizeof(int32_t) * E);
        assert_msg(n == static_cast<ssize_t>(sizeof(int32_t) * E),
                   "Failed to read GGR edge_dst: expected %zu bytes, got %zd",
                   sizeof(int32_t) * E, n);

        // int32 → uint32: 顶点 ID 非负，直接 reinterpret 即可
        // 但为了安全，验证范围
        int32_t* dest32 = reinterpret_cast<int32_t*>(csrResult.csr_dest);
        for (uint64_t i = 0; i < E; i++)
        {
            assert_msg(dest32[i] >= 0 && static_cast<uint64_t>(dest32[i]) < V,
                       "GGR: edge_dst[%lu] = %d out of range [0, %lu)", i, dest32[i], V);
        }
    }
    Msg_info("GGR: csr_dest loaded (%zu bytes)", sizeof(vertex_id_type) * E);

    // 6. 读取 weight（如有）
    if (hasWeight && needWeight)
    {
        // 跳过 padding（有权图且 E 为奇数时，4 字节对齐填充）
        if (E % 2 == 1)
        {
            off_t r = lseek(fd, 4, SEEK_CUR);
            assert_msg(r >= 0, "Failed to skip GGR padding");
        }

        csrResult.csr_weight = new edge_data_type[E];
        ssize_t n = read(fd, csrResult.csr_weight, sizeof(int32_t) * E);
        assert_msg(n == static_cast<ssize_t>(sizeof(int32_t) * E),
                   "Failed to read GGR adjwgt: expected %zu bytes, got %zd",
                   sizeof(int32_t) * E, n);
        Msg_info("GGR: csr_weight loaded (%zu bytes)", sizeof(edge_data_type) * E);
    }
    else if (needWeight)
    {
        // 无权图但算法需要权重 → 填充全 1
        csrResult.csr_weight = new edge_data_type[E];
        for (uint64_t i = 0; i < E; i++)
            csrResult.csr_weight[i] = 1;
        Msg_info("GGR: unweighted graph, filled csr_weight with 1");
    }
    else
    {
        csrResult.csr_weight = nullptr;
    }

    close(fd);
    Msg_info("GGR: load complete, |V|=%u, |E|=%u", csrResult.vertexNum, csrResult.edgeNum);
}

} // namespace CPJ
