#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>
#include <support/engines.hpp>

using namespace cudapfe;

namespace{
    constexpr std::size_t kRandomScalars = 500;
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
