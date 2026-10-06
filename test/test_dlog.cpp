#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>
#include <support/engines.hpp>

using namespace cudapfe;

namespace{
    using Exponents = std::vector<std::optional<std::int64_t>>;

    constexpr auto kMin = std::numeric_limits<std::int64_t>::min();
    constexpr auto kMax = std::numeric_limits<std::int64_t>::max();

    struct Batch{
        std::vector<Gt> targets;
        Exponents expected;
    };

    std::int64_t random_exponent(const std::int64_t lo, const std::int64_t hi){
        static std::mt19937_64 generator(20261005);
        return std::uniform_int_distribution<std::int64_t>(lo, hi)(generator);
    }

    Batch mixed_batch(const Gt& base, const Range& range, const std::size_t count){
        Batch batch;
        for (std::size_t i = 0; i < count; ++i){
            const auto k = i == 0 ? range.lo
                : i == 1 ? range.hi
                : i % 20 == 3 ? range.hi + 1 + random_exponent(0, 1000)
                : i % 20 == 13 ? range.lo - 1 - random_exponent(0, 1000)
                : random_exponent(range.lo, range.hi);
            batch.targets.push_back(base.pow(k));
            batch.expected.push_back(k < range.lo || k > range.hi ? std::nullopt : std::optional(k));
        }
        return batch;
    }
}

template <class E>
class DlogTest : public EngineTest<E>{};

TYPED_TEST_SUITE(DlogTest, Engines);

TYPED_TEST(DlogTest, FindsEveryExponentInItsInclusiveRange){
    const auto g = Gt::generator();
    const DlogTable<TypeParam> table(g, {-20, 20});
    std::vector<Gt> targets;
    Exponents expected;
    for (std::int64_t k = -21; k <= 21; ++k){
        targets.push_back(g.pow(k));
        expected.push_back(k < -20 || k > 20 ? std::nullopt : std::optional(k));
    }
    EXPECT_EQ(table.find(Vec<Gt, TypeParam>::upload(targets)), expected);
    EXPECT_EQ(DlogTable<TypeParam>(g, {-1000, 1000}).find(g.pow(777)), 777);
}

TYPED_TEST(DlogTest, HandlesExtremeRangesAndTheIdentityBase){
    const auto g = Gt::generator();
    EXPECT_EQ(DlogTable<TypeParam>(g, {kMax - 1, kMax}).find(g.pow(kMax)), kMax);
    EXPECT_EQ(DlogTable<TypeParam>(g, {kMax, kMax}).find(Gt()), std::nullopt);
    EXPECT_EQ(DlogTable<TypeParam>(g, {kMin, kMin + 100}).find(g.pow(kMin)), kMin);

    const DlogTable<TypeParam> trivial(Gt(), {5, 100});
    EXPECT_EQ(trivial.find(Gt()), 5);
    EXPECT_EQ(trivial.find(g), std::nullopt);
}

TYPED_TEST(DlogTest, FindsABatchWithMixedHitsAndMisses){
    const auto g = Gt::random();
    const Range range{0, 1 << 20};
    const auto batch = mixed_batch(g, range, 1000);
    EXPECT_EQ(DlogTable<TypeParam>(g, range).find(Vec<Gt, TypeParam>::upload(batch.targets)), batch.expected);
    EXPECT_TRUE(DlogTable<TypeParam>(g, range).find(Vec<Gt, TypeParam>()).empty());
}

TYPED_TEST(DlogTest, RejectsAnEmptyRange){
    const auto bases = Vec<Gt, TypeParam>::upload(std::vector<Gt>{Gt::generator()});
    for (const Range range : {Range{1, 0}, Range{kMax, kMin}}){
        EXPECT_THROW((void)DlogTable<TypeParam>(Gt::generator(), range), ShapeError) << range.lo;
        EXPECT_THROW((void)DlogTables<TypeParam>(bases, range), ShapeError) << range.lo;
    }
}

template <class E>
class DlogTablesTest : public EngineTest<E>{};

TYPED_TEST_SUITE(DlogTablesTest, Engines);

TYPED_TEST(DlogTablesTest, UsesOneBasePerTarget){
    const Range range{-1000, 1000};
    constexpr std::size_t count = 100;
    std::vector<Gt> bases;
    std::vector<Gt> targets;
    Exponents expected;
    for (std::size_t i = 0; i < count; ++i){
        const auto k = random_exponent(range.lo, range.hi);
        bases.push_back(i % 25 == 7 || i % 25 == 8 ? Gt() : Gt::random());
        targets.push_back(i % 25 == 8 ? Gt::generator() : bases.back().pow(k));
        expected.push_back(i % 25 == 7 ? std::optional(range.lo) : i % 25 == 8 ? std::nullopt : std::optional(k));
    }
    targets[10] = bases[11].pow(3);
    expected[10] = std::nullopt;

    const DlogTables<TypeParam> tables(Vec<Gt, TypeParam>::upload(bases), range);
    EXPECT_EQ(tables.find(Vec<Gt, TypeParam>::upload(targets)), expected);
}

TYPED_TEST(DlogTablesTest, RejectsMismatchedSizes){
    const auto bases = Vec<Gt, TypeParam>::upload(std::vector<Gt>(3, Gt::generator()));
    const DlogTables<TypeParam> tables(bases, {0, 10});
    EXPECT_THROW((void)tables.find(Vec<Gt, TypeParam>::upload(std::vector<Gt>(2))), ShapeError);
    EXPECT_THROW((void)tables.find(Vec<Gt, TypeParam>()), ShapeError);
    EXPECT_TRUE(DlogTables<TypeParam>(Vec<Gt, TypeParam>(), {0, 10}).find(Vec<Gt, TypeParam>()).empty());
}
