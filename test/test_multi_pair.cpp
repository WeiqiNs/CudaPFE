#include <concepts>
#include <cstddef>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>
#include <support/engines.hpp>
#include "pairing/plan.hpp"

using namespace cudapfe;

namespace{
    struct Pairs{
        std::vector<G1> ps;
        std::vector<G2> qs;
    };

    struct ExponentPairs{
        std::vector<Zp> a;
        std::vector<Zp> b;
    };

    struct Mismatch{
        std::string_view name;
        PairShape shape;
        std::size_t p_count;
        std::size_t q_count;
    };

    std::size_t side_count(const PairShape& shape, const Spread spread){
        return spread == Spread::shared ? shape.length : shape.segments * shape.length;
    }

    template <class P>
    std::vector<P> points_with_identities(const std::size_t count){
        constexpr std::size_t period = std::same_as<P, G1> ? 7 : 5;
        std::vector<P> points;
        for (std::size_t i = 0; i < count; ++i) points.push_back(i % period == 3 ? P() : P::random());
        return points;
    }

    Pairs random_pairs(const PairShape& shape){
        return {
            points_with_identities<G1>(side_count(shape, shape.p)),
            points_with_identities<G2>(side_count(shape, shape.q))
        };
    }

    std::vector<Gt> host_segments(const Pairs& pairs, const PairShape& shape){
        std::vector<Gt> results;
        for (std::size_t s = 0; s < shape.segments; ++s){
            std::vector<G1> ps;
            std::vector<G2> qs;
            for (std::size_t i = 0; i < shape.length; ++i){
                ps.push_back(pairs.ps[shape.p == Spread::shared ? i : s * shape.length + i]);
                qs.push_back(pairs.qs[shape.q == Spread::shared ? i : s * shape.length + i]);
            }
            results.push_back(pair(ps, qs));
        }
        return results;
    }

    std::vector<Zp> scalars_with_zeros(const std::size_t count){
        std::vector<Zp> scalars;
        for (std::size_t i = 0; i < count; ++i) scalars.push_back(i % 11 == 4 ? Zp() : Zp::random());
        return scalars;
    }

    std::vector<Gt> exponent_segments(const ExponentPairs& exponents, const PairShape& shape){
        std::vector<Gt> results;
        for (std::size_t s = 0; s < shape.segments; ++s){
            Zp exponent;
            for (std::size_t i = 0; i < shape.length; ++i){
                exponent += exponents.a[s * shape.length + i] * exponents.b[i];
            }
            results.push_back(Gt::generator().pow(exponent));
        }
        return results;
    }

    template <Engine E>
    std::vector<Gt> engine_segments(const Pairs& pairs, const PairShape& shape){
        return pair_segments(Vec<G1, E>::upload(pairs.ps), Vec<G2, E>::upload(pairs.qs), shape).download();
    }
}

template <class E>
class MultiPairTest : public EngineTest<E>{};

TYPED_TEST_SUITE(MultiPairTest, Engines);

TYPED_TEST(MultiPairTest, SegmentsMatchHostMultiPairing){
    const std::vector<PairShape> shapes{
        {1, 300}, {40, 7}, {40, 7, Spread::per_segment, Spread::shared}, {40, 7, Spread::shared, Spread::per_segment},
    };
    for (const auto& shape : shapes){
        const auto pairs = random_pairs(shape);
        EXPECT_EQ(engine_segments<TypeParam>(pairs, shape), host_segments(pairs, shape))
            << shape.segments << "x" << shape.length;
    }
}

TYPED_TEST(MultiPairTest, PreparedLinesAreReusable){
    const PairShape shape{40, 7, Spread::per_segment, Spread::shared};
    const auto first = random_pairs(shape);
    Pairs second{points_with_identities<G1>(side_count(shape, shape.p)), first.qs};
    const auto lines = prepare(Vec<G2, TypeParam>::upload(first.qs));

    EXPECT_EQ(lines.size(), shape.length);
    for (const auto& pairs : {first, second}){
        const auto ps = Vec<G1, TypeParam>::upload(pairs.ps);
        EXPECT_EQ(pair_segments(ps, lines, shape).download(), host_segments(pairs, shape));
    }
}

TYPED_TEST(MultiPairTest, RejectsMismatchedShapes){
    constexpr auto shared = Spread::shared;
    constexpr auto per_segment = Spread::per_segment;
    const std::vector<Mismatch> cases{
        {"zero length", {2, 0}, 0, 0},
        {"short G1 side", {2, 3}, 5, 6},
        {"long G2 side", {2, 3}, 6, 7},
        {"shared G1 sized per segment", {2, 3, shared, per_segment}, 6, 6},
        {"shared G2 sized per segment", {2, 3, per_segment, shared}, 6, 6},
        {"overflowing shape", {std::numeric_limits<std::size_t>::max(), 2}, 0, 0},
    };
    for (const auto& [name, shape, p_count, q_count] : cases){
        const auto ps = Vec<G1, TypeParam>::upload(std::vector<G1>(p_count));
        const auto qs = Vec<G2, TypeParam>::upload(std::vector<G2>(q_count));
        EXPECT_THROW((void)pair_segments(ps, qs, shape), ShapeError) << name;
        EXPECT_THROW((void)pair_segments(ps, prepare(qs), shape), ShapeError) << name;
    }

    const auto ps = Vec<G1, TypeParam>::upload(std::vector<G1>(6));
    const auto qs = Vec<G2, TypeParam>::upload(std::vector<G2>(6));
    const auto ones = pair_segments(ps, qs, PairShape{2, 3}).download();
    EXPECT_EQ(ones, std::vector<Gt>(2));
}

TYPED_TEST(MultiPairTest, EmptyShapeGivesEmptyResult){
    const std::vector<PairShape> shapes{
        {0, 3}, {0, 3, Spread::shared, Spread::per_segment}, {0, 3, Spread::per_segment, Spread::shared},
    };
    for (const auto& shape : shapes){
        const auto pairs = random_pairs(shape);
        const auto ps = Vec<G1, TypeParam>::upload(pairs.ps);
        const auto qs = Vec<G2, TypeParam>::upload(pairs.qs);
        EXPECT_TRUE(pair_segments(ps, qs, shape).empty());
        EXPECT_TRUE(pair_segments(ps, prepare(qs), shape).empty());
    }
}

TEST(MultiPairTest, ItemsSpanSeveralPairsOnLargeBatches){
    CUDAPFE_REQUIRE_GPU();
    const PairShape shape{64, 2701, Spread::per_segment, Spread::shared};
    const ExponentPairs exponents{scalars_with_zeros(shape.segments * shape.length), scalars_with_zeros(shape.length)};
    const auto ps = mul_generator<G1>(Vec<Zp, Gpu>::upload(exponents.a));
    const auto qs = mul_generator<G2>(Vec<Zp, Gpu>::upload(exponents.b));

    EXPECT_EQ(pair_segments(ps, qs, shape).download(), exponent_segments(exponents, shape));
}

TEST(MultiPairTest, EveryPlacementAgrees){
    CUDAPFE_REQUIRE_GPU();
    using detail::kHostFinalExpBelow;
    using detail::kHostMillerBelow;
    using detail::Placement;
    if (kHostMillerBelow < 2 || kHostFinalExpBelow < 2) GTEST_SKIP() << "thresholds leave a placement unreachable";
    const std::vector<std::pair<Placement, PairShape>> cases{
        {Placement::host, {1, kHostMillerBelow - 1}},
        {Placement::host_final_exp, {1, kHostMillerBelow}},
        {Placement::device, {kHostFinalExpBelow, detail::ceil_div(kHostMillerBelow, kHostFinalExpBelow)}},
    };
    for (const auto& [placement, shape] : cases){
        ASSERT_EQ(detail::place(shape), placement);
        const auto pairs = random_pairs(shape);
        const auto expected = host_segments(pairs, shape);
        const auto ps = Vec<G1, Gpu>::upload(pairs.ps);
        const auto qs = Vec<G2, Gpu>::upload(pairs.qs);
        EXPECT_EQ(pair_segments(ps, qs, shape).download(), expected) << static_cast<int>(placement);
        EXPECT_EQ(pair_segments(ps, prepare(qs), shape).download(), expected) << static_cast<int>(placement);
    }
}
