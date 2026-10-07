#include <array>
#include <cstddef>
#include <span>
#include <vector>
#include <support/oracle.hpp>
#include <support/samples.hpp>
#include "curve/curve.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "pairing/miller.hpp"

using namespace cudapfe;
using namespace cudapfe::detail;
using namespace cudapfe::test;

namespace{
    constexpr std::size_t kLineSamples = 20;
    constexpr std::size_t kMultiPairs = 5;
    constexpr std::size_t kFinalExpSamples = 20;

    template <class F>
    std::vector<Affine<F>> affine_points(const std::size_t count){
        std::vector<Affine<F>> points;
        for (const auto& p : random_points<F>(count)){
            if (!p.is_identity()) points.push_back(to_affine(p));
        }
        return points;
    }

    std::vector<Line> host_lines(const std::span<const G2Affine> qs){
        std::vector<Line> lines(qs.size() * kLineCount);
        for (std::size_t i = 0; i < qs.size(); ++i) prepare_lines(qs[i], lines.data() + i * kLineCount);
        return lines;
    }

    Fp12 host_miller(const std::span<const G1Affine> ps, const std::span<const G2Affine> qs){
        const auto lines = host_lines(qs);
        return miller(PreparedPairs{ps.data(), qs.data(), lines.data(), ps.size()});
    }

    Fp12 blst_miller(const std::span<const G1Affine> ps, const std::span<const G2Affine> qs){
        std::vector<Blst<G1Affine>> blst_ps;
        std::vector<Blst<G2Affine>> blst_qs;
        for (std::size_t i = 0; i < ps.size(); ++i){
            blst_ps.push_back(to_blst(ps[i]));
            blst_qs.push_back(to_blst(qs[i]));
        }
        std::vector<const Blst<G1Affine>*> p_pointers;
        std::vector<const Blst<G2Affine>*> q_pointers;
        for (std::size_t i = 0; i < ps.size(); ++i){
            p_pointers.push_back(&blst_ps[i]);
            q_pointers.push_back(&blst_qs[i]);
        }
        blst_fp12 result;
        blst_miller_loop_n(&result, q_pointers.data(), p_pointers.data(), ps.size());
        return from_blst<Fp12>(result);
    }

    std::vector<Fp12> final_exp_inputs(){
        auto inputs = tower_samples<Fp12>(kFinalExpSamples);
        const auto ps = affine_points<Fp>(kFinalExpSamples);
        const auto qs = affine_points<Fp2>(kFinalExpSamples);
        for (std::size_t i = 0; i < kFinalExpSamples; ++i) inputs.push_back(host_miller({&ps[i], 1}, {&qs[i], 1}));
        return inputs;
    }
}

TEST(MillerTest, LinesMatchBlst){
    for (const auto& q : affine_points<Fp2>(kLineSamples)){
        const auto lines = host_lines({&q, 1});
        std::array<blst_fp6, kLineCount> expected;
        const auto blst_q = to_blst(q);
        blst_precompute_lines(expected.data(), &blst_q);
        for (std::size_t step = 0; step < kLineCount; ++step){
            EXPECT_EQ(lines[step], from_blst<Fp6>(expected[step])) << step;
        }
    }
}

TEST(MillerTest, MillerLoopMatchesBlst){
    const auto ps = affine_points<Fp>(kMultiPairs);
    const auto qs = affine_points<Fp2>(kMultiPairs);
    const auto blst_p = to_blst(ps.front());
    const auto blst_q = to_blst(qs.front());
    blst_fp12 single;
    blst_miller_loop(&single, &blst_q, &blst_p);

    EXPECT_EQ(host_miller({ps.data(), 1}, {qs.data(), 1}), from_blst<Fp12>(single));
    EXPECT_EQ(host_miller(std::span(ps).first(kMultiPairs), std::span(qs).first(kMultiPairs)),
        blst_miller(std::span(ps).first(kMultiPairs), std::span(qs).first(kMultiPairs)));
}

TEST(MillerTest, FinalExpMatchesBlst){
    for (const auto& f : final_exp_inputs()){
        EXPECT_EQ(final_exp(f), from_blst<Fp12>(blst_result(blst_final_exp, to_blst(f))));
    }
}
