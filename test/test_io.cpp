#include <format>
#include <sstream>
#include <string>
#include <gtest/gtest.h>
#include <cufe/cufe.hpp>

using namespace cufe;

TEST(IoTest, HexIsTwoLowercaseDigitsPerByte){
    EXPECT_EQ(to_hex(Bytes{0x00, 0xab, 0x10, 0xff}), "00ab10ff");
    EXPECT_EQ(to_hex(Bytes{}), "");
}

TEST(FormatTest, ScalarsPrintInDecimal){
    EXPECT_EQ(std::format("{}", Zp(42)), "42");
    EXPECT_EQ(std::format("{:>4}", Zp(7)), "   7");

    std::ostringstream out;
    out << Zp(1234);
    EXPECT_EQ(out.str(), "1234");
}

TEST(FormatTest, GroupElementsPrintTheirEncodingInHex){
    EXPECT_EQ(std::format("{}", G1()), "c0" + std::string(94, '0'));
    EXPECT_EQ(std::format("{}", G2()), "c0" + std::string(190, '0'));

    const auto generator = std::format("{}", G1::generator());
    EXPECT_EQ(generator.size(), 2 * G1::compressed_size);
    EXPECT_TRUE(generator.starts_with("97f1d3a73197d794"));

    const auto g = Gt::generator();
    const auto gt = std::format("{}", g);
    EXPECT_EQ(gt.size(), 2 * Gt::byte_size);

    std::ostringstream out;
    out << G1() << ' ' << G2::generator() << ' ' << g;
    EXPECT_EQ(out.str(), std::format("{}", G1()) + " " + std::format("{}", G2::generator()) + " " + gt);
}
