#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <support/engines.hpp>
#include <support/oracle.hpp>
#include <support/samples.hpp>
#include "curve/curve.hpp"
#include "curve/straus.hpp"
#include "field/constants.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "support/words.hpp"

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
    struct StrausCase{
        std::vector<Affine<F>> bases;
        std::vector<Fr> scalars;
        std::size_t stride = 1;
        std::size_t bits_at_least = 0;
    };

    template <class F>
    struct EvaluateStraus{
        const MsmTerms<F>* terms;
        Affine<F>* sums;

        CUDAPFE_HD void operator()(const std::size_t i) const{ sums[i] = to_affine(straus_sum(terms[i])); }
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

    std::vector<Fr> edge_and_random_scalars(){
        const auto one = Fr::one();
        const auto z = Fr::from_canonical({kZ});
        const auto z2 = z * z;
        const auto two64 = Fr::from_canonical({0, 1});
        const auto two128 = Fr::from_canonical({0, 0, 1});
        std::vector<Fr> scalars{
            Fr::zero(), one, one + one, Fr::from_canonical({15}), Fr::from_canonical({16}), z - one, z, z + one, z2 - one,
            z2, z2 + one, z2 * z, two64 - one, two128 - one, two128, Fr::from_canonical(sub_word(kR, 2)),
            Fr::from_canonical(sub_word(kR, 1))
        };
        for (const auto& k : field_samples<FrParams>(kRandomPoints)) scalars.push_back(k);
        return scalars;
    }

    template <class F>
    std::vector<CurveInput<F>> curve_inputs(){
        const auto points = random_points<F>(kRandomPoints);
        const auto scalars = edge_and_random_scalars();
        std::vector<CurveInput<F>> inputs;
        for (std::size_t i = 0; i < points.size(); ++i){
            const auto& p = points[i];
            const auto normalized = from_affine(to_affine(p));
            const auto& next = points[(i + 1) % points.size()];
            const auto q = i % 4 == 1 ? normalized : i % 4 == 2 ? neg(normalized) : next;
            inputs.push_back({p, q, scalars[i]});
        }
        return inputs;
    }

    Fr small_fr(const std::int64_t k){
        const auto magnitude = Fr::from_canonical({static_cast<Word>(k < 0 ? -k : k)});
        return k < 0 ? -magnitude : magnitude;
    }

    Fr low_bits_set(const std::size_t count){
        Words<Fr::N> words{};
        for (std::size_t i = 0; i < count; ++i) words[i / 64] |= Word{1} << (i % 64);
        return Fr::from_canonical(words);
    }

    template <class F>
    std::vector<StrausCase<F>> straus_cases(){
        const auto points = random_points<F>(2);
        const auto p = to_affine(points[3]);
        const auto q = to_affine(points[4]);
        const auto r_minus_one = Fr::from_canonical(sub_word(kR, 1));
        return {
            {{p, q}, {small_fr(3), small_fr(5)}},
            {{p, p}, {small_fr(1), small_fr(1)}},
            {{p, p, q}, {small_fr(1), small_fr(-1), small_fr(1)}},
            {{Affine<F>::identity(), p}, {small_fr(7), small_fr(2)}},
            {{p, q}, {Fr::zero(), Fr::zero()}},
            {{p, q, p}, {small_fr((1ll << 31) + 1), small_fr(-((1ll << 30) + 3)), low_bits_set(kStrausMaxBits)}},
            {{p, q}, {small_fr(3), small_fr(5)}, 1, 64},
            {{p, q}, {small_fr(3), r_minus_one, r_minus_one, small_fr(5)}, 3},
        };
    }

    template <class F>
    struct StrausInputs{
        std::vector<Affine<F>> bases;
        std::vector<SignedScalar> scalars;
    };

    template <class F>
    StrausInputs<F> flattened(const std::vector<StrausCase<F>>& cases){
        StrausInputs<F> inputs;
        for (const auto& c : cases){
            inputs.bases.insert(inputs.bases.end(), c.bases.begin(), c.bases.end());
            for (const auto& k : c.scalars) inputs.scalars.push_back(signed_scalar(k));
        }
        return inputs;
    }

    template <class F>
    std::vector<MsmTerms<F>> straus_terms(const std::vector<StrausCase<F>>& cases, const Affine<F>* bases,
        const SignedScalar* scalars){
        std::vector<MsmTerms<F>> terms;
        for (const auto& c : cases){
            auto bits = c.bits_at_least;
            for (std::size_t k = 0; k < c.bases.size(); ++k){
                bits = std::max(bits, bit_length(signed_scalar(c.scalars[k * c.stride]).magnitude));
            }
            terms.push_back({bases, scalars, c.stride, c.bases.size(), bits});
            bases += c.bases.size();
            scalars += c.scalars.size();
        }
        return terms;
    }

    template <class F>
    Affine<F> blst_straus(const StrausCase<F>& c){
        auto sum = to_blst(Jacobian<F>::identity());
        for (std::size_t k = 0; k < c.bases.size(); ++k){
            if (c.bases[k].is_identity()) continue;
            const auto term = blst_mult<F>(to_blst(from_affine(c.bases[k])), c.scalars[k * c.stride]);
            sum = blst_result(CurveOracle<F>::add, sum, term);
        }
        return blst_affine<F>(sum);
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
    using O = CurveOracle<F>;
    for (std::uint8_t x = 1; x <= 40; ++x){
        Blst<Affine<F>> point;
        if (O::uncompress(&point, small_x_compressed<F>(x).data()) != BLST_SUCCESS) continue;
        EXPECT_EQ(in_subgroup(from_blst<Affine<F>>(point)), O::in_group(&point)) << int{x};
    }
    EXPECT_TRUE(in_subgroup(to_affine(random_points<F>(1).back())));
}

TYPED_TEST(CurveTest, BatchToAffineMatchesSingle){
    using F = TypeParam;
    auto points = random_points<F>(kBatchPoints);
    for (std::size_t i = 0; i < points.size(); i += 7) points[i] = Jacobian<F>::identity();
    std::vector<Affine<F>> batch(points.size());
    to_affine(points.data(), points.size(), batch.data());
    for (std::size_t i = 0; i < points.size(); ++i) EXPECT_EQ(batch[i], to_affine(points[i])) << i;
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

TYPED_TEST(CurveTest, StrausSumMatchesBlst){
    using F = TypeParam;
    const auto cases = straus_cases<F>();
    const auto inputs = flattened(cases);
    const auto terms = straus_terms(cases, inputs.bases.data(), inputs.scalars.data());
    for (std::size_t i = 0; i < cases.size(); ++i){
        EXPECT_EQ(to_affine(straus_sum(terms[i])), blst_straus(cases[i])) << i;
    }
}

TYPED_TEST(CurveTest, StrausSumOnDeviceMatchesHost){
    CUDAPFE_REQUIRE_GPU();
    using F = TypeParam;
    const auto cases = straus_cases<F>();
    const auto inputs = flattened(cases);
    DeviceArray<Affine<F>> bases(inputs.bases.size());
    DeviceArray<SignedScalar> scalars(inputs.scalars.size());
    DeviceArray<MsmTerms<F>> terms(cases.size());
    DeviceArray<Affine<F>> sums(cases.size());
    bases.copy_from(inputs.bases);
    scalars.copy_from(inputs.scalars);
    terms.copy_from(straus_terms(cases, bases.data(), scalars.data()));
    for_each<Gpu>(cases.size(), EvaluateStraus<F>{terms.data(), sums.data()});

    const auto actual = sums.copy_to_host();
    const auto host_terms = straus_terms(cases, inputs.bases.data(), inputs.scalars.data());
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], to_affine(straus_sum(host_terms[i]))) << i;
}
