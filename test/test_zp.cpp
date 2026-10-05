#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <gtest/gtest.h>
#include <cufe/cufe.hpp>

using namespace cufe;

namespace{
    constexpr std::string_view kOrder = "52435875175126190479447740508185965837690552500527637822603658699938581184513";

    std::string decrement(std::string decimal){
        EXPECT_NE(decimal.back(), '0');
        --decimal.back();
        return decimal;
    }
}

TEST(ZpTest, ArithmeticAgreesWithIntegers){
    EXPECT_EQ(Zp(7) + Zp(5), Zp(12));
    EXPECT_EQ(Zp(7) - Zp(12), Zp(-5));
    EXPECT_EQ(Zp(6) * Zp(7), Zp(42));
    EXPECT_EQ(Zp(42) / Zp(7), Zp(6));
    EXPECT_EQ(-Zp(3), Zp(-3));
    EXPECT_EQ(-Zp(), Zp());
    EXPECT_EQ(Zp(3).pow(5), Zp(243));
    EXPECT_EQ(Zp(3).pow(0), Zp(1));
    EXPECT_EQ(Zp(5u), Zp(5));
    static_assert(!std::is_constructible_v<Zp, bool> && !std::is_constructible_v<Zp, char>);
    static_assert(!std::is_constructible_v<Zp, char8_t> && !std::is_constructible_v<Zp, wchar_t>);
    static_assert(std::is_constructible_v<Zp, std::int8_t> && std::is_constructible_v<Zp, std::uint8_t>);

    Zp x = 10;
    x += 4;
    x -= 2;
    x *= 3;
    x /= 4;
    EXPECT_EQ(x, Zp(9));
}

TEST(ZpTest, ValuesReduceModuloTheGroupOrder){
    EXPECT_EQ(Zp(-1).to_string(), decrement(std::string(kOrder)));
    EXPECT_TRUE((Zp(-1) + 1).is_zero());
    EXPECT_FALSE(Zp(1).is_zero());
    EXPECT_EQ(Zp().to_string(), "0");
    EXPECT_EQ(Zp(std::numeric_limits<std::uint64_t>::max()).to_string(), "18446744073709551615");
    EXPECT_TRUE((Zp(std::numeric_limits<std::int64_t>::min()) + Zp(std::uint64_t{1} << 63)).is_zero());
}

TEST(ZpTest, InverseExistsExceptForZero){
    const auto x = Zp::random();
    EXPECT_EQ(x * x.inverse(), Zp(1));
    EXPECT_THROW((void)Zp().inverse(), NotInvertible);
    EXPECT_THROW((void)(Zp(1) / Zp(0)), NotInvertible);
}

TEST(ZpTest, EncodingIsFixedWidthBigEndianAndValidated){
    const auto bytes = Zp(258).to_bytes();
    ASSERT_EQ(bytes.size(), Zp::byte_size);
    EXPECT_EQ(bytes.at(bytes.size() - 2), 0x01);
    EXPECT_EQ(bytes.back(), 0x02);
    EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end() - 2, [](const auto b){ return b == 0; }));
    EXPECT_EQ(Zp::from_bytes(bytes), Zp(258));

    const auto largest = Zp(-1).to_bytes();
    EXPECT_EQ(Zp::from_bytes(largest), Zp(-1));

    auto order = largest;
    ++order.back();
    EXPECT_THROW((void)Zp::from_bytes(order), DecodeError);

    const Bytes short_by_one(largest.begin() + 1, largest.end());
    EXPECT_THROW((void)Zp::from_bytes(short_by_one), DecodeError);
}
