#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/pairing.hpp>
#include "pairing/plan.hpp"
#include "vec/reduction.hpp"

using namespace cudapfe;
using namespace cudapfe::detail;

namespace{
    struct PairsPerItemCase{
        PairShape shape;
        std::size_t resident_threads;
        std::size_t expected;
    };

    struct ReductionCase{
        std::size_t width;
        std::size_t levels;
    };
}

TEST(PlanTest, ItemsCoverEveryPairOnceWithoutCrossingSegments){
    for (const auto& shape : std::vector<PairShape>{{1, 1}, {1, 300}, {50, 7}, {3, 8}, {5, 17}}){
        for (const std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{8}, shape.length}){
            const ItemPlan plan{shape, k};
            std::vector<int> visits(shape.segments * shape.length, 0);
            for (std::size_t j = 0; j < plan.item_count(); ++j){
                const auto item = plan.item(j);
                ASSERT_EQ(item.segment, j / plan.items_per_segment()) << j;
                ASSERT_GE(item.count, 1u) << j;
                ASSERT_LE(item.count, k) << j;
                ASSERT_LE(item.first + item.count, shape.length) << j;
                for (std::size_t i = 0; i < item.count; ++i) ++visits[item.segment * shape.length + item.first + i];
            }
            EXPECT_EQ(visits, std::vector<int>(visits.size(), 1))
                << shape.segments << "x" << shape.length << " k=" << k;
        }
    }
}

TEST(PlanTest, ReductionReachesOneProductPerSegment){
    constexpr std::size_t segments = 3;
    for (const auto& [width, expected_levels] : std::vector<ReductionCase>{{1, 0}, {2, 1}, {7, 2}, {300, 5}}){
        Reduction level{segments, width};
        std::size_t levels = 0;
        while (level.width > 1){
            std::vector<int> visits(segments * level.width, 0);
            for (std::size_t g = 0; g < level.group_count(); ++g){
                const auto group = level.group(g);
                const auto segment = g / level.groups_per_segment();
                ASSERT_GE(group.count, 1u) << g;
                ASSERT_LE(group.count, kReduceFanIn) << g;
                ASSERT_GE(group.first, segment * level.width) << g;
                ASSERT_LE(group.first + group.count, (segment + 1) * level.width) << g;
                for (std::size_t i = 0; i < group.count; ++i) ++visits[group.first + i];
            }
            EXPECT_EQ(visits, std::vector<int>(visits.size(), 1)) << "width " << level.width;
            level = level.next();
            ++levels;
        }
        EXPECT_EQ(levels, expected_levels) << "width " << width;
        EXPECT_EQ(level.group_count(), segments) << "width " << width;
    }
}

TEST(PlanTest, PairsPerItemClampsToBounds){
    const std::vector<PairsPerItemCase> cases{
        {{1, 10}, 1000, 1},
        {{100, 50}, 1000, 5},
        {{1, 100000}, 1000, kMaxPairsPerItem},
        {{0, 5}, 1000, 1},
        {{4, 4}, 0, kMaxPairsPerItem},
    };
    for (const auto& [shape, resident_threads, expected] : cases){
        EXPECT_EQ(pairs_per_item(shape, resident_threads), expected) << shape.segments << "x" << shape.length;
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
