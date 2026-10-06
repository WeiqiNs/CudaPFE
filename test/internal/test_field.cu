#include <algorithm>
#include <cstddef>
#include <vector>
#include <support/engines.hpp>
#include <support/oracle.hpp>
#include <support/samples.hpp>
#include "field/constants.hpp"
#include "field/field.hpp"
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"

using namespace cudapfe;
using namespace cudapfe::detail;
using namespace cudapfe::test;

namespace{
    constexpr std::size_t kRandomSamples = 1000;

    template <class F>
    struct Results{
        F sum, difference, product, square, inverse, negation, power, round_trip;
        Words<F::N> canonical;
        bool equal, zero;

        friend bool operator==(const Results&, const Results&) = default;
    };

    template <class F>
    CUDAPFE_HD Results<F> evaluate(const F& x, const F& y){
        constexpr auto exponent = kFpSqrtExponent;
        return {
            x + y, x - y, x * y, x.square(), x.inverse(), -x, x.pow(exponent), F::from_canonical(x.canonical()),
            x.canonical(), x == y, x.is_zero()
        };
    }

    template <class F>
    struct Evaluate{
        const F* xs;
        const F* ys;
        Results<F>* results;

        CUDAPFE_HD void operator()(const std::size_t i) const{ results[i] = evaluate(xs[i], ys[i]); }
    };

    template <class F>
    struct InvertOddSquareEven{
        const F* xs;
        F* results;

        CUDAPFE_HD void operator()(const std::size_t i) const{
            if (i % 2 == 1) results[i] = xs[i].inverse();
            else results[i] = xs[i].square();
        }
    };

    template <class F>
    std::vector<F> partners(const std::vector<F>& xs){
        std::vector<F> ys(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) ys[i] = i % 5 == 0 ? xs[i] : xs[(i + 1) % xs.size()];
        return ys;
    }
}

template <class Params>
class FieldTest : public ::testing::Test{};

using FieldParams = ::testing::Types<FpParams, FrParams>;
TYPED_TEST_SUITE(FieldTest, FieldParams);

TYPED_TEST(FieldTest, HostMatchesBlst){
    using F = Field<TypeParam>;
    using O = Oracle<TypeParam>;
    const auto xs = field_samples<TypeParam>(kRandomSamples);
    const auto ys = partners(xs);

    for (std::size_t i = 0; i < xs.size(); ++i){
        const auto& x = xs[i];
        const auto& y = ys[i];
        const auto a = to_blst(x);
        const auto b = to_blst(y);
        EXPECT_EQ(x + y, from_blst<F>(blst_result(O::add, a, b))) << i;
        EXPECT_EQ(x - y, from_blst<F>(blst_result(O::sub, a, b))) << i;
        EXPECT_EQ(x * y, from_blst<F>(blst_result(O::mul, a, b))) << i;
        EXPECT_EQ(x.square(), from_blst<F>(blst_result(O::sqr, a))) << i;
        EXPECT_EQ(x.inverse(), from_blst<F>(blst_result(O::inverse, a))) << i;
        EXPECT_EQ(x.pow(TypeParam::inverse_exponent), from_blst<F>(blst_result(O::inverse, a))) << i;
        EXPECT_EQ(-x, from_blst<F>(blst_result(O::neg, a))) << i;
    }
    EXPECT_EQ(F::zero().inverse(), F::zero());
}

TYPED_TEST(FieldTest, CanonicalRoundTrips){
    using F = Field<TypeParam>;
    for (const auto& words : canonical_samples<TypeParam>(kRandomSamples)){
        const auto x = F::from_canonical(words);
        EXPECT_EQ(x.canonical(), words);
        EXPECT_EQ(x.montgomery(), blst_montgomery<TypeParam>(words));
        EXPECT_EQ(F::from_montgomery(x.montgomery()), x);
    }
}

TYPED_TEST(FieldTest, DeviceMatchesHost){
    CUDAPFE_REQUIRE_GPU();
    using F = Field<TypeParam>;
    const auto xs = field_samples<TypeParam>(kRandomSamples);
    const auto ys = partners(xs);
    std::vector<Results<F>> expected;
    for (std::size_t i = 0; i < xs.size(); ++i) expected.push_back(evaluate(xs[i], ys[i]));

    DeviceArray<F> device_xs(xs.size());
    DeviceArray<F> device_ys(ys.size());
    DeviceArray<Results<F>> results(xs.size());
    device_xs.copy_from(xs);
    device_ys.copy_from(ys);
    for_each<Gpu>(xs.size(), Evaluate<F>{device_xs.data(), device_ys.data(), results.data()});

    const auto actual = results.copy_to_host();
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], expected[i]) << i;
    EXPECT_TRUE(std::any_of(expected.begin(), expected.end(), [](const auto& r){ return r.equal; }));
}

TYPED_TEST(FieldTest, InverseWorksInDivergentThreads){
    CUDAPFE_REQUIRE_GPU();
    using F = Field<TypeParam>;
    const auto xs = field_samples<TypeParam>(kRandomSamples);
    DeviceArray<F> device_xs(xs.size());
    DeviceArray<F> results(xs.size());
    device_xs.copy_from(xs);
    for_each<Gpu>(xs.size(), InvertOddSquareEven<F>{device_xs.data(), results.data()});

    const auto actual = results.copy_to_host();
    for (std::size_t i = 0; i < xs.size(); ++i){
        EXPECT_EQ(actual[i], i % 2 == 1 ? xs[i].inverse() : xs[i].square()) << i;
    }
}

TEST(FieldTest, ConstantsMatchVendoredSppark){
#ifndef __CUDA_ARCH__
    EXPECT_TRUE(std::equal(kP.begin(), kP.end(), bls12_381::BLS12_381_P));
    EXPECT_TRUE(std::equal(kR.begin(), kR.end(), bls12_381::BLS12_381_r));
#endif
    EXPECT_EQ(Fp::one().canonical(), (Words<6>{1}));
    EXPECT_EQ(Fr::one().canonical(), (Words<4>{1}));
}
