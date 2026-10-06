#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <blst.h>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>
#include <support/engines.hpp>

using namespace cudapfe;

namespace{
    constexpr std::size_t kRandomScalars = 500;
    constexpr std::size_t kOracleScalars = 20;
    constexpr unsigned kTopWindowShift = 248;
    constexpr unsigned kLargestTopDigit = 0x73;

    std::vector<Zp> edge_and_random_scalars(){
        std::vector<Zp> scalars{Zp(), Zp(1), Zp(2), Zp(-1)};
        for (std::uint64_t k = 0; k < 255; ++k) scalars.push_back(Zp(2).pow(k));
        const auto top_window = Zp(2).pow(kTopWindowShift);
        for (unsigned digit = 0; digit <= kLargestTopDigit; ++digit) scalars.push_back(Zp(digit) * top_window);
        for (std::size_t i = 0; i < kRandomScalars; ++i) scalars.push_back(Zp::random());
        return scalars;
    }

    Bytes little_endian(const Zp& k){
        auto bytes = k.to_bytes();
        std::reverse(bytes.begin(), bytes.end());
        return bytes;
    }
}

template <class C>
class MulGeneratorTest : public CaseTest<C>{};

using MulGeneratorCases = ::testing::Types<Case<G1, Cpu>, Case<G1, Gpu>, Case<G2, Cpu>, Case<G2, Gpu>>;
TYPED_TEST_SUITE(MulGeneratorTest, MulGeneratorCases);

TYPED_TEST(MulGeneratorTest, MatchesVariableBaseMultiplication){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    const auto scalars = edge_and_random_scalars();
    const auto points = mul_generator<P>(Vec<Zp, E>::upload(scalars)).download();

    ASSERT_EQ(points.size(), scalars.size());
    for (std::size_t i = 0; i < scalars.size(); ++i) EXPECT_EQ(points[i], P::generator() * scalars[i]) << i;
    EXPECT_TRUE(mul_generator<P>(Vec<Zp, E>()).empty());
}

TEST(MulGeneratorTest, HostSingleMatchesBlst){
    for (std::size_t i = 0; i < kOracleScalars; ++i){
        const auto k = i == 0 ? Zp(-1) : Zp::random();
        const auto scalar = little_endian(k);
        blst_p1 p1;
        blst_p2 p2;
        blst_p1_mult(&p1, blst_p1_generator(), scalar.data(), 255);
        blst_p2_mult(&p2, blst_p2_generator(), scalar.data(), 255);
        Bytes expected_g1(G1::compressed_size), expected_g2(G2::compressed_size);
        blst_p1_compress(expected_g1.data(), &p1);
        blst_p2_compress(expected_g2.data(), &p2);

        EXPECT_EQ(G1::mul_generator(k).to_bytes(), expected_g1) << i;
        EXPECT_EQ(G2::mul_generator(k).to_bytes(), expected_g2) << i;
    }
}
