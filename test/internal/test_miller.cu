#include <array>
#include <cstddef>
#include <span>
#include <vector>
#include <support/engines.hpp>
#include <support/oracle.hpp>
#include <support/samples.hpp>
#include "curve/curve.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "pairing/miller.hpp"
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"

using namespace cufe;
using namespace cufe::detail;
using namespace cufe::test;

namespace{
    constexpr std::size_t kLineSamples = 20;
    constexpr std::size_t kMultiPairs = 5;
    constexpr std::size_t kFinalExpSamples = 20;
    constexpr std::size_t kTextbookSamples = 3;
    constexpr std::size_t kDeviceItems = 4;
    constexpr std::size_t kPairsPerItem = 3;

    constexpr Words<68> kTextbookExponent{
        0xc0bcb9b55df57510, 0x25f98630e68bfb24, 0x4406fbc8fbd5f489, 0x8e2f8491d12191a0, 0x3e9d71650a6f8069,
        0x226c2f011d4cab80, 0x67f67c4717489119, 0xaf3f881bd88592d7, 0x1a67e49eeed2161d, 0xe5b78c7869aeb218,
        0xf6539314043f7bbc, 0x73f62537f2701aae, 0xaff1c910e9622d2a, 0x6283313492caa9d4, 0x2e2f3ec2bea83d19,
        0xa4c7e79fb02faa73, 0x6c49637fd7961be1, 0x08e88adce8817745, 0x35de3f7a36399917, 0x9c1d9f7c31759c36,
        0xfa9e13c24ea820b0, 0x3fc56947a403577d, 0xa4c1b6dcfc5cceb7, 0x1bbd81367066bca6, 0x0418a3ef0bc62775,
        0x49bf9b71a9f9e010, 0x511291097db60b17, 0x498345c6e5308f1c, 0x6d8823b19dadd7c2, 0x92004cedd556952c,
        0x4c6bec3ec03ef195, 0x0a1fad20044ce6ad, 0xc55d3109cd15948d, 0x334f46c02c3f0bd0, 0x3b5a62eb34c05739,
        0x724538411d1676a5, 0x127a1b5ad0463434, 0x61a474c5c85b0129, 0x8dfc8e2886ef965e, 0x96532fef459f1243,
        0x40ee7169cdc10412, 0x9c40a68eb74bb22a, 0x25118790f4684d0b, 0x596bc293c8d4c01f, 0x1064837f27611212,
        0x077ffb10bf24dde4, 0xc49f570bcd2b01f3, 0x1a0c5bf24c374693, 0x350da5359bc73ab6, 0xd2670d93e4d7acdd,
        0xd39099b86e1ab656, 0x19328148978e2b0d, 0xb113f414386b0e88, 0x07a0dce2630d9aa4, 0xa927e7bb93753318,
        0xe347aa68ad49466f, 0x1c0ad0d6106feaf4, 0xc872ee83ff3a0f0f, 0x074e43b9a660835c, 0xc0aadff5e9cfee9a,
        0x30698e8cc7deada9, 0xd1073776ab353f2c, 0x17848517badc3a43, 0x7363baa13f8d14a9, 0xd4977b3f7d4507d0,
        0x496a1c0a89ee0193, 0xdcc825b7e1bda9c0, 0x0000000002ee1db5
    };

    struct PrepareLinesOp{
        const G2Affine* qs;
        Line* lines;

        CUFE_HD void operator()(const std::size_t i) const{ prepare_lines(qs[i], lines + i * kLineCount); }
    };

    struct MillerOp{
        const G1Affine* px2s;
        const Line* lines;
        Fp12* results;

        CUFE_HD void operator()(const std::size_t i) const{
            const auto first = i * kPairsPerItem;
            results[i] = miller(PreparedPairs{px2s + first, lines + first * kLineCount, kPairsPerItem});
        }
    };

    struct FinalExpOp{
        const Fp12* values;
        Fp12* results;

        CUFE_HD void operator()(const std::size_t i) const{ results[i] = final_exp(values[i]); }
    };

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

    std::vector<G1Affine> host_px2s(const std::span<const G1Affine> ps){
        std::vector<G1Affine> px2s;
        for (const auto& p : ps) px2s.push_back(px2(p));
        return px2s;
    }

    Fp12 host_miller(const std::span<const G1Affine> ps, const std::span<const G2Affine> qs){
        const auto px2s = host_px2s(ps);
        const auto lines = host_lines(qs);
        return miller(PreparedPairs{px2s.data(), lines.data(), ps.size()});
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

TEST(MillerTest, FinalExpIsTheCubeOfTheTextbookPower){
    const auto inputs = final_exp_inputs();
    for (std::size_t i = 0; i < kTextbookSamples; ++i){
        for (const auto& f : {inputs[i], inputs[kFinalExpSamples + i]}){
            EXPECT_EQ(final_exp(f), pow(pow(f, kTextbookExponent), Words<1>{3}));
        }
    }
}

TEST(MillerTest, DeviceMatchesHost){
    CUFE_REQUIRE_GPU();
    constexpr std::size_t pairs = kDeviceItems * kPairsPerItem;
    const auto ps = affine_points<Fp>(pairs);
    const auto qs = affine_points<Fp2>(pairs);
    const auto px2s = host_px2s(std::span(ps).first(pairs));
    const auto lines = host_lines(std::span(qs).first(pairs));

    DeviceArray<G2Affine> device_qs(pairs);
    DeviceArray<G1Affine> device_px2s(pairs);
    DeviceArray<Line> device_lines(lines.size());
    DeviceArray<Fp12> device_millers(kDeviceItems);
    DeviceArray<Fp12> device_finals(kDeviceItems);
    device_qs.copy_from(std::span(qs).first(pairs));
    device_px2s.copy_from(px2s);
    for_each<Gpu>(pairs, PrepareLinesOp{device_qs.data(), device_lines.data()});
    for_each<Gpu>(kDeviceItems, MillerOp{device_px2s.data(), device_lines.data(), device_millers.data()});
    for_each<Gpu>(kDeviceItems, FinalExpOp{device_millers.data(), device_finals.data()});

    EXPECT_EQ(device_lines.copy_to_host(), lines);
    const auto millers = device_millers.copy_to_host();
    const auto finals = device_finals.copy_to_host();
    for (std::size_t item = 0; item < kDeviceItems; ++item){
        const auto first = item * kPairsPerItem;
        const PreparedPairs item_pairs{px2s.data() + first, lines.data() + first * kLineCount, kPairsPerItem};
        const auto expected = miller(item_pairs);
        EXPECT_EQ(millers[item], expected) << item;
        EXPECT_EQ(finals[item], final_exp(expected)) << item;
    }
}
