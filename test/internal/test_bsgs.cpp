#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <cufe/dlog.hpp>
#include "dlog/bsgs.hpp"

using namespace cufe;
using namespace cufe::detail;

namespace{
    constexpr auto kMin = std::numeric_limits<std::int64_t>::min();
    constexpr auto kMax = std::numeric_limits<std::int64_t>::max();

    using Wide = unsigned __int128;

    struct StepsCase{
        Range range;
        std::uint64_t span;
        std::uint64_t baby;
        std::uint64_t giant;
    };

    struct LadderCase{
        std::size_t entries;
        std::uint64_t length;
        std::size_t resident_threads;
        std::uint64_t per_ladder;
    };
}

TEST(BsgsPlanTest, StepsCoverTheRange){
    const std::vector<StepsCase> cases{
        {{0, 0}, 0, 1, 1},
        {{kMax, kMax}, 0, 1, 1},
        {{kMax - 1, kMax}, 1, 2, 1},
        {{-20, 20}, 40, 7, 6},
        {{kMin, kMin + 100}, 100, 11, 10},
        {{0, 1 << 20}, 1 << 20, 1025, 1024},
        {{0, std::int64_t{1} << 32}, std::uint64_t{1} << 32, 65537, 65536},
        {{kMin, kMax}, std::numeric_limits<std::uint64_t>::max(), std::uint64_t{1} << 32, std::uint64_t{1} << 32},
    };
    for (const auto& [range, span, baby, giant] : cases){
        const auto steps = plan_steps(range);
        EXPECT_EQ(steps.lo, range.lo) << range.lo;
        EXPECT_EQ(steps.span, span) << range.lo;
        EXPECT_EQ(steps.baby, baby) << range.lo;
        EXPECT_EQ(steps.giant, giant) << range.lo;
        EXPECT_GT(Wide{steps.baby} * steps.giant, Wide{steps.span}) << range.lo;
        EXPECT_LE(Wide{steps.baby} * (steps.giant - 1), Wide{steps.span}) << range.lo;
        EXPECT_LE(Wide{steps.baby} * steps.giant, Wide{1} << 64) << range.lo;
        EXPECT_EQ(offset(steps.lo, steps.span), range.hi) << range.lo;
    }
    EXPECT_THROW((void)plan_steps({1, 0}), ShapeError);
    EXPECT_THROW((void)plan_steps({kMax, kMin}), ShapeError);
}

TEST(BsgsPlanTest, LaddersPartitionEachEntry){
    const std::vector<LadderCase> cases{
        {1, 1, 1000000, 1},
        {1, 1025, 1000, 2},
        {50, 7, 1000, 1},
        {3, 1025, 1, 1025},
        {3, 1025, 10, 308},
        {0, 7, 1000, 1},
    };
    for (const auto& [entries, length, resident_threads, per_ladder] : cases){
        const auto plan = plan_ladders(entries, length, resident_threads);
        EXPECT_EQ(plan.per_ladder, per_ladder) << entries << "x" << length;
        std::vector<int> visits(entries * length, 0);
        for (std::size_t l = 0; l < plan.ladder_count(); ++l){
            const auto ladder = plan.ladder(l);
            ASSERT_LT(ladder.entry, entries) << l;
            ASSERT_GE(ladder.count, 1u) << l;
            ASSERT_LE(ladder.count, per_ladder) << l;
            ASSERT_LE(ladder.first + ladder.count, length) << l;
            for (std::uint64_t i = 0; i < ladder.count; ++i) ++visits[ladder.entry * length + ladder.first + i];
        }
        EXPECT_EQ(visits, std::vector<int>(visits.size(), 1)) << entries << "x" << length;
    }
}

TEST(BsgsPlanTest, ChunksRespectTheBudget){
    for (const std::uint64_t baby : {std::uint64_t{1}, std::uint64_t{7}, std::uint64_t{65537}, std::uint64_t{1} << 32}){
        for (const std::size_t budget : {kTableBudgetBytes, std::size_t{1000}, std::size_t{1}}){
            const auto chunk = entries_per_chunk(baby, budget);
            const auto entry_bytes = Wide{baby} * kBabyStepBytes;
            if (entry_bytes > budget){
                EXPECT_EQ(chunk, 1u) << baby << " " << budget;
            } else {
                EXPECT_LE(chunk * entry_bytes, budget) << baby << " " << budget;
                EXPECT_GT((chunk + 1) * entry_bytes, budget) << baby << " " << budget;
            }
        }
    }
}
