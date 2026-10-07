#include <cstddef>
#include <vector>
#include <support/engines.hpp>
#include <support/oracle.hpp>
#include <support/samples.hpp>
#include "curve/curve.hpp"
#include "field/constants.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"

using namespace cudapfe;
using namespace cudapfe::detail;
using namespace cudapfe::test;

namespace{
    constexpr std::size_t kRandomPoints = 100;
    constexpr std::size_t kBatchPoints = 50;

    template <class F>
    struct CurveInput{
        Jacobian<F> p, q;
        Fr k;
    };

    template <class F>
    struct CurveResults{
        Affine<F> sum, mixed_sum, twice, negation, product, affine;

        friend bool operator==(const CurveResults&, const CurveResults&) = default;
    };

    template <class F>
    CUDAPFE_HD CurveResults<F> evaluate(const CurveInput<F>& in){
        return {
            to_affine(add(in.p, in.q)), to_affine(add_mixed(in.p, to_affine(in.q))), to_affine(dbl(in.p)),
            to_affine(neg(in.p)), to_affine(mul(in.p, in.k)), to_affine(in.p)
        };
    }

    template <class F>
    struct EvaluateCurve{
        const CurveInput<F>* inputs;
        CurveResults<F>* results;

        CUDAPFE_HD void operator()(const std::size_t i) const{ results[i] = evaluate(inputs[i]); }
    };

    template <class F>
    CurveResults<F> blst_evaluate(const CurveInput<F>& in){
        using O = CurveOracle<F>;
        const auto p = to_blst(in.p);
        const auto q = to_blst(in.q);
        auto negation = p;
        O::cneg(&negation, true);
        return {
            blst_affine<F>(blst_result(O::add, p, q)),
            blst_affine<F>(blst_result(O::add_affine, p, blst_result(O::to_affine, q))),
            blst_affine<F>(blst_result(O::dbl, p)),
            blst_affine<F>(negation),
            blst_affine<F>(blst_mult<F>(p, in.k)),
            blst_affine<F>(p)
        };
    }

    template <class F>
    std::vector<CurveInput<F>> curve_inputs(){
        const auto points = random_points<F>(kRandomPoints);
        const auto scalars = field_samples<FrParams>(kRandomPoints + 1);
        std::vector<CurveInput<F>> inputs;
        for (std::size_t i = 0; i < points.size(); ++i){
            const auto& p = points[i];
            const auto normalized = from_affine(to_affine(p));
            const auto& next = points[(i + 1) % points.size()];
            const auto q = i % 4 == 1 ? normalized : i % 4 == 2 ? neg(normalized) : next;
            inputs.push_back({p, q, scalars[i + 1]});
        }
        return inputs;
    }
}

template <class F>
class CurveTest : public ::testing::Test{};

TYPED_TEST_SUITE(CurveTest, Sides, SideNames);

TYPED_TEST(CurveTest, HostMatchesBlst){
    const auto inputs = curve_inputs<TypeParam>();
    for (std::size_t i = 0; i < inputs.size(); ++i){
        const auto actual = evaluate(inputs[i]);
        const auto expected = blst_evaluate(inputs[i]);
        EXPECT_EQ(actual.sum, expected.sum) << i;
        EXPECT_EQ(actual.mixed_sum, expected.mixed_sum) << i;
        EXPECT_EQ(actual.twice, expected.twice) << i;
        EXPECT_EQ(actual.negation, expected.negation) << i;
        EXPECT_EQ(actual.product, expected.product) << i;
        EXPECT_EQ(actual.affine, expected.affine) << i;
    }
}

TYPED_TEST(CurveTest, EdgeCasesFollowTheGroupLaw){
    using F = TypeParam;
    const auto p = random_points<F>(1).back();
    const auto identity = Jacobian<F>::identity();
    const auto r_minus_one = Fr::from_canonical(sub_word(kR, 1));

    EXPECT_TRUE(equal(add(p, identity), p));
    EXPECT_TRUE(equal(add(identity, p), p));
    EXPECT_TRUE(equal(add(p, p), dbl(p)));
    EXPECT_TRUE(add(p, neg(p)).is_identity());
    EXPECT_TRUE(equal(add_mixed(p, to_affine(p)), dbl(p)));
    EXPECT_TRUE(add_mixed(p, to_affine(neg(p))).is_identity());
    EXPECT_TRUE(equal(add_mixed(identity, to_affine(p)), p));
    EXPECT_TRUE(mul(p, Fr::zero()).is_identity());
    EXPECT_TRUE(equal(mul(p, Fr::one()), p));
    EXPECT_TRUE(equal(mul(p, r_minus_one), neg(p)));
    EXPECT_TRUE(mul(p, kR).is_identity());
    EXPECT_FALSE(equal(p, identity));
    EXPECT_EQ(CurveParams<F>::generator(), blst_affine<F>(*CurveOracle<F>::generator()));
}

TYPED_TEST(CurveTest, SubgroupCheckAgreesWithBlst){
    using F = TypeParam;
    const auto outside = first_point_outside_subgroup<F>();
    EXPECT_TRUE(on_curve(outside));
    EXPECT_FALSE(in_subgroup(outside));

    const auto inside = to_affine(random_points<F>(1).back());
    const auto blst_inside = to_blst(inside);
    EXPECT_TRUE(CurveOracle<F>::in_group(&blst_inside));
    EXPECT_TRUE(on_curve(inside));
    EXPECT_TRUE(in_subgroup(inside));
    EXPECT_TRUE(in_subgroup(CurveParams<F>::generator()));
}

TYPED_TEST(CurveTest, BatchToAffineMatchesSingle){
    using F = TypeParam;
    auto points = random_points<F>(kBatchPoints);
    for (std::size_t i = 0; i < points.size(); i += 7) points[i] = Jacobian<F>::identity();
    std::vector<Affine<F>> batch(points.size());
    to_affine(points.data(), points.size(), batch.data());
    for (std::size_t i = 0; i < points.size(); ++i) EXPECT_EQ(batch[i], to_affine(points[i])) << i;

    const std::vector identities(10, Jacobian<F>::identity());
    to_affine(identities.data(), identities.size(), batch.data());
    for (std::size_t i = 0; i < identities.size(); ++i) EXPECT_EQ(batch[i], Affine<F>::identity()) << i;
}

TYPED_TEST(CurveTest, DeviceMatchesHost){
    CUDAPFE_REQUIRE_GPU();
    using F = TypeParam;
    const auto inputs = curve_inputs<F>();
    DeviceArray<CurveInput<F>> device_inputs(inputs.size());
    DeviceArray<CurveResults<F>> results(inputs.size());
    device_inputs.copy_from(inputs);
    for_each<Gpu>(inputs.size(), EvaluateCurve<F>{device_inputs.data(), results.data()});

    const auto actual = results.copy_to_host();
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], evaluate(inputs[i])) << i;
}
