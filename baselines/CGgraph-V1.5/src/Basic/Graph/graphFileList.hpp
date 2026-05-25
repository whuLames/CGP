#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/compute_degree.hpp"
#include "Basic/Graph/getGraphCSR.hpp"
#include "Basic/Graph/sortNbr.hpp"
#include "Basic/IO/io_ggr.hpp"
#include "Basic/Other/fileSystem_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include "flag.hpp"
#include "reorderGraph.hpp"
#include <assert.h>
#include <string>

namespace CPJ {

/* *********************************************************************************************************
 * @description: 按需修改图数据路径
 * @param [string&] graphName
 * @return [*]
 * *********************************************************************************************************/
vertex_id_type* getGraphData(CSR_Result_type& csrResult, std::string graphName, bool isSortNbr, bool getOutDegree, bool getInDegree)
{
    vertex_id_type* old2new{nullptr};
    GraphFile_type graphFile;

    // GGR 格式检测: 当 graphName 是一个存在的 .ggr/.gr 文件时走此分支
    if (CPJ::FS::isExist(graphName) && isGGRFile(graphName))
    {
        std::string displayName = CPJ::FS::getFileStem(graphName);
        Algorithm_type algorithm = static_cast<Algorithm_type>(SCI32(FLAGS_algorithm));

        // 检查 reorder 缓存是否已存在
        if (CPJ::FS::isExist(getCGgraph_reorder_rankFile(displayName)))
        {
            GGR_Header header = readGGRHeader(graphName);
            graphFile.vertices = header.nvtxs;
            graphFile.edges = header.nedges;
            graphFile.rankFile = getCGgraph_reorder_rankFile(displayName);
            graphFile.old2newFile = getCGgraph_reorder_old2newFile(displayName);
            graphFile.csrOffsetFile = getCGgraph_reorder_csrOffset(displayName);
            graphFile.csrDestFile = getCGgraph_reorder_csrDest(displayName);
            graphFile.csrWeightFile = getCGgraph_reorder_csrWeight(displayName);

            old2new = CPJ::getGraphCSR(csrResult, graphFile, algorithm, OrderMethod_type::CGgraphRV1_5,
                                       isSortNbr, getOutDegree, getInDegree);
        }
        else
        {
            // 首次运行: 从 GGR 加载原始 CSR，然后执行 reorder
            // reorder 始终需要 weight，所以 needWeight=true
            CSR_Result_type csrResult_native;
            CPJ::loadGGR(graphName, csrResult_native, true);

            // sortNbr
            if (isSortNbr)
                CPJ::sortNbr(csrResult_native, true, false);

            // 计算 degree (reorder 需要 outDegree，所以始终计算)
            Compute_degree computeDegree(csrResult_native);
            csrResult_native.outDegree = computeDegree.getOutdegree();
            if (getInDegree) csrResult_native.inDegree = computeDegree.getIndegree();

            // 执行 reorder
            CPJ::ReorderGraph reorderGraph(csrResult_native);
            csrResult = reorderGraph.doReorder(displayName);
            reorderGraph.freeOldCSR();
            old2new = reorderGraph.getOld2New();
        }

        return old2new;
    }

    // 以下是原有的硬编码 graphName 分支
    if (graphName == "test")
    {
        graphFile.vertices = 6;
        graphFile.edges = 15;

        graphFile.csrOffsetFile = "/home/zyl/Projects/ocgp/baselines/CGgraph-V1.5/test_data/native_csrOffset_u32.bin";
        graphFile.csrDestFile = "/home/zyl/Projects/ocgp/baselines/CGgraph-V1.5/test_data/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/home/zyl/Projects/ocgp/baselines/CGgraph-V1.5/test_data/native_csrWeight_u32.bin";

    } // end of [test]

    else if (graphName == "twitter2010")
    {
        graphFile.vertices = 61578415;
        graphFile.edges = 1468364884;

        graphFile.csrOffsetFile = "/data/webgraph/bin/twitter2010/native_csrOffset_u32.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/twitter2010/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/twitter2010/native_csrWeight_u32.bin";

    } // end of [twitter2010]

    else if (graphName == "friendster")
    {
        graphFile.vertices = 124836180;
        graphFile.edges = 1806067135;

        graphFile.csrOffsetFile = "/data/webgraph/bin/friendster/native_csrOffset_u32.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/friendster/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/friendster/native_csrWeight_u32.bin";
    }
    else if (graphName == "uk-union")
    {
        graphFile.vertices = 133633040;
        graphFile.edges = 5475109924;

        graphFile.csrOffsetFile = "/data/webgraph/bin/uk-union/native_csrOffset_u64.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/uk-union/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/uk-union/native_csrWeight_u32.bin";
    }
    else
    {
        assert_msg(false, "Unknow graphName [%s]", graphName.c_str());
    }

    assert_msg(graphFile.vertices < std::numeric_limits<count_type>::max(),
               "Total vertices need set the <count_type> and <vertex_id_type> to uint64_t");
    assert_msg(graphFile.edges < std::numeric_limits<countl_type>::max(), "Total edges need set the <countl_type> to uint64_t");
    if (graphFile.edges < std::numeric_limits<uint32_t>::max())
    {
        bool isSame = std::is_same_v<countl_type, uint32_t>;
        assert_msg(isSame, "Total edges can be stored by uint32_t, So set the <countl_type> to uint32_t");
    }

    if (CPJ::FS::isExist(getCGgraph_reorder_rankFile(graphName)))
    {
        graphFile.rankFile = getCGgraph_reorder_rankFile(graphName);
        graphFile.old2newFile = getCGgraph_reorder_old2newFile(graphName);
        graphFile.csrOffsetFile = getCGgraph_reorder_csrOffset(graphName);
        graphFile.csrDestFile = getCGgraph_reorder_csrDest(graphName);
        graphFile.csrWeightFile = getCGgraph_reorder_csrWeight(graphName);

        old2new = CPJ::getGraphCSR(csrResult, graphFile, static_cast<Algorithm_type>(SCI32(FLAGS_algorithm)), OrderMethod_type::CGgraphRV1_5,
                                   isSortNbr, getOutDegree, getInDegree);
    }
    else
    {
        CSR_Result_type csrResult_native;
        CPJ::getGraphCSR(csrResult_native, graphFile, static_cast<Algorithm_type>(SCI32(FLAGS_algorithm)), OrderMethod_type::NATIVE, isSortNbr, true,
                         getInDegree);
        CPJ::ReorderGraph reorderGraph(csrResult_native);
        csrResult = reorderGraph.doReorder(graphName);
        reorderGraph.freeOldCSR();
        old2new = reorderGraph.getOld2New();
    }

    return old2new;
}

} // namespace CPJ
