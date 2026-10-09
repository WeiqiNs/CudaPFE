#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/pairing.hpp>
#include "pairing/plan.hpp"
#include "vec/reduction.hpp"

using namespace cudapfe;
using namespace cudapfe::detail;

namespace{
    struct ChunkSizeCase{
        std::size_t total;
        std::size_t threads;
        std::size_t cap;
        std::size_t expected;
    };
}

TEST(PlanTest, ChunksCoverEachSegmentOnce){
    for (const auto& shape : std::vector<PairShape>{{1, 1}, {1, 300}, {50, 7}, {3, 8}, {5, 17}, {0, 7}}){
        for (const std::size_t size : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}, shape.length}){
            const Chunks chunks{shape.segments, shape.length, size};
            std::vector<int> visits(shape.segments * shape.length, 0);
            for (std::size_t j = 0; j < chunks.count(); ++j){
                const auto chunk = chunks.at(j);
                ASSERT_LT(chunk.segment, shape.segments) << j;
                ASSERT_GE(chunk.count, 1u) << j;
                ASSERT_LE(chunk.count, size) << j;
                ASSERT_LE(chunk.first + chunk.count, shape.length) << j;
                for (std::size_t i = 0; i < chunk.count; ++i) ++visits[chunk.segment * shape.length + chunk.first + i];
            }
            EXPECT_EQ(visits, std::vector<int>(visits.size(), 1))
                << shape.segments << "x" << shape.length << " size " << size;
        }
    }
}

TEST(PlanTest, ChunkSizeClampsToBounds){
    const std::vector<ChunkSizeCase> cases{
        {0, 1000, kMaxPairsPerItem, 1},
        {5001, 1000, kMaxPairsPerItem, 6},
        {100000, 1000, kMaxPairsPerItem, kMaxPairsPerItem},
        {16, 0, kMaxPairsPerItem, kMaxPairsPerItem},
    };
    for (const auto& [total, threads, cap, expected] : cases){
        EXPECT_EQ(chunk_size(total, threads, cap), expected) << total << " over " << threads << " up to " << cap;
    }
}

TEST(PlanTest, PlacementUsesThresholds){
    if (kHostMillerBelow < 2 || kHostFinalExpBelow < 2) GTEST_SKIP() << "thresholds leave a placement unreachable";
    const auto device_length = ceil_div(kHostMillerBelow, kHostFinalExpBelow);

    EXPECT_EQ(place({1, kHostMillerBelow - 1}), Placement::host);
    EXPECT_EQ(place({kHostMillerBelow - 1, 1}), Placement::host);
    EXPECT_EQ(place({1, kHostMillerBelow}), Placement::host_final_exp);
    EXPECT_EQ(place({kHostFinalExpBelow - 1, kHostMillerBelow}), Placement::host_final_exp);
    EXPECT_EQ(place({kHostFinalExpBelow, device_length}), Placement::device);
    EXPECT_EQ(place({kHostFinalExpBelow, device_length, Spread::shared, Spread::shared}), Placement::device);
}
