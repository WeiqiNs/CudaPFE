#include <cstddef>
#include <vector>
#include <support/engines.hpp>
#include <support/oracle.hpp>
#include <support/samples.hpp>
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
    constexpr std::size_t kSamples = 200;
    constexpr std::size_t kPowerSamples = 20;

    struct TowerInput{
        Fp2 a, b;
        Fp12 x, y, unitary;
        Fp6 line;
    };

    struct TowerResults{
        Fp2 fp2_product, fp2_square, fp2_inverse;
        Fp12 product, square, conjugate, inverse, frobenius1, frobenius2, frobenius3, line_product, cyclotomic_square;
    };

    CUDAPFE_HD TowerResults evaluate(const TowerInput& in){
        return {
            in.a * in.b, in.a.square(), in.a.inverse(),
            in.x * in.y, in.x.square(), conjugate(in.x), in.x.inverse(),
            frobenius<1>(in.x), frobenius<2>(in.x), frobenius<3>(in.x),
            mul_by_line(in.x, in.line), cyclotomic_square(in.unitary)
        };
    }

    struct EvaluateTower{
        const TowerInput* inputs;
        TowerResults* results;

        CUDAPFE_HD void operator()(const std::size_t i) const{ results[i] = evaluate(inputs[i]); }
    };

    TowerResults blst_evaluate(const TowerInput& in){
        const auto a = to_blst(in.a);
        const auto b = to_blst(in.b);
        const auto x = to_blst(in.x);
        return {
            from_blst<Fp2>(blst_result(blst_fp2_mul, a, b)),
            from_blst<Fp2>(blst_result(blst_fp2_sqr, a)),
            from_blst<Fp2>(blst_result(blst_fp2_inverse, a)),
            from_blst<Fp12>(blst_result(blst_fp12_mul, x, to_blst(in.y))),
            from_blst<Fp12>(blst_result(blst_fp12_sqr, x)),
            from_blst<Fp12>(blst_conjugate(x)),
            from_blst<Fp12>(blst_result(blst_fp12_inverse, x)),
            from_blst<Fp12>(blst_frobenius(x, 1)),
            from_blst<Fp12>(blst_frobenius(x, 2)),
            from_blst<Fp12>(blst_frobenius(x, 3)),
            from_blst<Fp12>(blst_result(blst_fp12_mul_by_xy00z0, x, to_blst(in.line))),
            from_blst<Fp12>(blst_result(blst_fp12_cyclotomic_sqr, to_blst(in.unitary)))
        };
    }

    void expect_same(const TowerResults& actual, const TowerResults& expected, const std::size_t i){
        EXPECT_EQ(actual.fp2_product, expected.fp2_product) << i;
        EXPECT_EQ(actual.fp2_square, expected.fp2_square) << i;
        EXPECT_EQ(actual.fp2_inverse, expected.fp2_inverse) << i;
        EXPECT_EQ(actual.product, expected.product) << i;
        EXPECT_EQ(actual.square, expected.square) << i;
        EXPECT_EQ(actual.conjugate, expected.conjugate) << i;
        EXPECT_EQ(actual.inverse, expected.inverse) << i;
        EXPECT_EQ(actual.frobenius1, expected.frobenius1) << i;
        EXPECT_EQ(actual.frobenius2, expected.frobenius2) << i;
        EXPECT_EQ(actual.frobenius3, expected.frobenius3) << i;
        EXPECT_EQ(actual.line_product, expected.line_product) << i;
        EXPECT_EQ(actual.cyclotomic_square, expected.cyclotomic_square) << i;
    }

    std::vector<TowerInput> tower_inputs(){
        const auto fp2s = tower_samples<Fp2>(2 * kSamples);
        const auto fp12s = tower_samples<Fp12>(2 * kSamples);
        const auto lines = tower_samples<Fp6>(kSamples);
        std::vector<TowerInput> inputs;
        for (std::size_t i = 0; i < kSamples; ++i){
            const auto& x = fp12s[2 * i];
            const auto unitary = from_blst<Fp12>(blst_result(blst_final_exp, to_blst(x)));
            inputs.push_back({fp2s[2 * i], fp2s[2 * i + 1], x, fp12s[2 * i + 1], unitary, lines[i]});
        }
        return inputs;
    }
}

TEST(TowerTest, HostMatchesBlst){
    const auto inputs = tower_inputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) expect_same(evaluate(inputs[i]), blst_evaluate(inputs[i]), i);
}

TEST(TowerTest, FrobeniusIsThePthPower){
    for (const auto& x : tower_samples<Fp12>(kPowerSamples)) EXPECT_EQ(frobenius<1>(x), pow(x, kP));
}

TEST(TowerTest, CyclotomicPowMatchesPow){
    const std::vector<Words<4>> exponents{
        Words<4>{}, Words<4>{1}, Words<4>{2}, sub_word(kR, 1), kR,
        Words<4>{0x9e3779b97f4a7c15, 0xf39cc0605cedc834, 0x1082276bf3a27251, 0x7a7ac1b4c8a3d4e5}
    };
    for (const auto& input : tower_inputs()){
        for (const auto& e : exponents) EXPECT_EQ(cyclotomic_pow(input.unitary, e), pow(input.unitary, e));
    }
}

TEST(TowerTest, DeviceMatchesHost){
    CUDAPFE_REQUIRE_GPU();
    const auto inputs = tower_inputs();
    DeviceArray<TowerInput> device_inputs(inputs.size());
    DeviceArray<TowerResults> results(inputs.size());
    device_inputs.copy_from(inputs);
    for_each<Gpu>(inputs.size(), EvaluateTower{device_inputs.data(), results.data()});

    const auto actual = results.copy_to_host();
    for (std::size_t i = 0; i < actual.size(); ++i) expect_same(actual[i], evaluate(inputs[i]), i);
}
