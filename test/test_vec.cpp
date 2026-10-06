#include <cstddef>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>
#include <support/engines.hpp>

using namespace cudapfe;

namespace{
    constexpr std::size_t kRoundTripValues = 1000;
    constexpr std::size_t kElementwiseValues = 300;

    template <class T>
    std::vector<T> values_with_neutral_ends(const std::size_t count){
        std::vector<T> values{T()};
        while (values.size() + 1 < count) values.push_back(T::random());
        values.push_back(T());
        return values;
    }

    template <class P>
    std::vector<P> partners(const std::vector<P>& x){
        std::vector<P> y;
        for (std::size_t i = 0; i < x.size(); ++i){
            switch (i % 5){
            case 1: y.push_back(x[i]); break;
            case 2: y.push_back(-x[i]); break;
            case 3: y.push_back(P()); break;
            default: y.push_back(P::random());
            }
        }
        return y;
    }

    std::vector<Zp> scalars(const std::size_t count){
        std::vector<Zp> k{Zp(), Zp(1), Zp(-1)};
        while (k.size() < count) k.push_back(Zp::random());
        return k;
    }
}

template <class C>
class VecTest : public CaseTest<C>{};

using VecCases = ::testing::Types<
    Case<Zp, Cpu>, Case<Zp, Gpu>, Case<G1, Cpu>, Case<G1, Gpu>, Case<G2, Cpu>, Case<G2, Gpu>, Case<Gt, Cpu>,
    Case<Gt, Gpu>>;
TYPED_TEST_SUITE(VecTest, VecCases);

TYPED_TEST(VecTest, UploadDownloadRoundTrips){
    using T = typename TypeParam::Value;
    using V = Vec<T, typename TypeParam::Engine>;
    const auto values = values_with_neutral_ends<T>(kRoundTripValues);
    const auto uploaded = V::upload(values);

    EXPECT_EQ(uploaded.size(), kRoundTripValues);
    EXPECT_EQ(uploaded.download(), values);
    EXPECT_EQ(uploaded.at(0), T());
    EXPECT_EQ(uploaded.at(1), values[1]);
    EXPECT_EQ(uploaded.at(kRoundTripValues - 1), T());
    EXPECT_THROW((void)uploaded.at(kRoundTripValues), std::out_of_range);

    const auto empty = V::upload({});
    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(empty.download().empty());
    EXPECT_THROW((void)empty.at(0), std::out_of_range);
    EXPECT_TRUE(V().empty());
}

template <class C>
class PointVecTest : public CaseTest<C>{};

using PointCases = ::testing::Types<Case<G1, Cpu>, Case<G1, Gpu>, Case<G2, Cpu>, Case<G2, Gpu>>;
TYPED_TEST_SUITE(PointVecTest, PointCases);

TYPED_TEST(PointVecTest, ElementwiseOpsMatchHost){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    const auto x = values_with_neutral_ends<P>(kElementwiseValues);
    const auto y = partners(x);
    const auto k = scalars(kElementwiseValues);
    const auto vx = Vec<P, E>::upload(x);
    const auto vy = Vec<P, E>::upload(y);
    const auto vk = Vec<Zp, E>::upload(k);

    const auto sums = (vx + vy).download();
    const auto differences = (vx - vy).download();
    const auto negations = (-vx).download();
    const auto products = (vx * vk).download();
    for (std::size_t i = 0; i < x.size(); ++i){
        EXPECT_EQ(sums[i], x[i] + y[i]) << i;
        EXPECT_EQ(differences[i], x[i] - y[i]) << i;
        EXPECT_EQ(negations[i], -x[i]) << i;
        EXPECT_EQ(products[i], x[i] * k[i]) << i;
    }
}

TYPED_TEST(PointVecTest, RejectsMismatchedSizes){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    const auto three = Vec<P, E>::upload(std::vector{P::random(), P(), P::random()});
    const auto two = Vec<P, E>::upload(std::vector{P::random(), P::random()});
    const auto scalars = Vec<Zp, E>::upload(std::vector{Zp(2), Zp(3)});

    EXPECT_THROW((void)(three + two), ShapeError);
    EXPECT_THROW((void)(three - two), ShapeError);
    EXPECT_THROW((void)(three * scalars), ShapeError);
}

TYPED_TEST(PointVecTest, ConcatJoinsTheSegmentsOfEveryPart){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    using V = Vec<P, E>;
    constexpr std::size_t segments = 4;
    constexpr std::size_t length = 3;
    const auto own = values_with_neutral_ends<P>(segments * length);
    const std::vector shared{P::random(), P::random()};
    const auto single = values_with_neutral_ends<P>(segments);
    std::vector<P> expected;
    for (std::size_t s = 0; s < segments; ++s){
        expected.insert(expected.end(), own.begin() + s * length, own.begin() + (s + 1) * length);
        expected.insert(expected.end(), shared.begin(), shared.end());
        expected.push_back(single[s]);
    }

    const auto joined = concat<P, E>(segments, {{V::upload(own), length}, {V::upload(shared), 2, Spread::shared},
        {V::upload(single), 1}});
    EXPECT_EQ(joined.download(), expected);
    EXPECT_TRUE((concat<P, E>(0, {{V(), length}, {V::upload(shared), 2, Spread::shared}}).empty()));
    EXPECT_THROW((void)(concat<P, E>(segments, {{V::upload(shared), 1}})), ShapeError);
}

template <class E>
class GtVecTest : public EngineTest<E>{};

TYPED_TEST_SUITE(GtVecTest, Engines);

TYPED_TEST(GtVecTest, ElementwiseProductAndQuotient){
    using V = Vec<Gt, TypeParam>;
    const auto x = values_with_neutral_ends<Gt>(kElementwiseValues);
    auto y = values_with_neutral_ends<Gt>(kElementwiseValues);
    y[1] = x[1];
    y[2] = x[2].inverse();
    const auto vx = V::upload(x);
    const auto vy = V::upload(y);

    const auto products = (vx * vy).download();
    const auto quotients = (vx / vy).download();
    for (std::size_t i = 0; i < x.size(); ++i){
        EXPECT_EQ(products[i], x[i] * y[i]) << i;
        EXPECT_EQ(quotients[i], x[i] / y[i]) << i;
    }
    EXPECT_TRUE(quotients[1].is_one());
    EXPECT_TRUE(products[2].is_one());

    const auto shorter = V::upload(std::vector{Gt::random()});
    EXPECT_THROW((void)(vx * shorter), ShapeError);
    EXPECT_THROW((void)(vx / shorter), ShapeError);
}
