#include <cstddef>
#include <string_view>
#include <vector>
#include <blst.h>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>

using namespace cudapfe;

namespace{
    constexpr std::size_t kFpBytes = 48;

    struct Candidate{
        std::string_view name;
        Bytes bytes;
        bool in_group;
    };

    blst_fp12 blst_from_bytes(const Bytes& bytes){
        blst_fp12 f;
        const auto* next = bytes.data();
        for (std::size_t slot = 0; slot < 3; ++slot){
            for (std::size_t half = 0; half < 2; ++half){
                for (auto& coefficient : f.fp6[half].fp2[slot].fp){
                    blst_fp_from_bendian(&coefficient, next);
                    next += kFpBytes;
                }
            }
        }
        return f;
    }

    Bytes blst_bytes(const blst_fp12& f){
        Bytes out(Gt::byte_size);
        blst_bendian_from_fp12(out.data(), &f);
        return out;
    }

    blst_fp12 easy_part(const blst_fp12& f){
        blst_fp12 conjugate = f, inverse, unitary, frobenius;
        blst_fp12_conjugate(&conjugate);
        blst_fp12_inverse(&inverse, &f);
        blst_fp12_mul(&unitary, &conjugate, &inverse);
        blst_fp12_frobenius_map(&frobenius, &unitary, 2);
        blst_fp12_mul(&unitary, &unitary, &frobenius);
        return unitary;
    }

    bool decodes(const Bytes& bytes){
        try{
            (void)Gt::from_bytes(bytes);
            return true;
        } catch (const DecodeError&){
            return false;
        }
    }
}

TEST(GtTest, GroupLawsHold){
    const auto x = Gt::random(), y = Gt::random();

    EXPECT_EQ(x * Gt(), x);
    EXPECT_TRUE((x / x).is_one());
    EXPECT_TRUE((x * x.inverse()).is_one());
    EXPECT_EQ(x * y, y * x);
    EXPECT_FALSE(x.is_one());

    auto acc = x;
    acc *= y;
    acc /= x;
    EXPECT_EQ(acc, y);
}

TEST(GtTest, PowAcceptsNegativeExponents){
    const auto g = Gt::generator();

    EXPECT_EQ(g.pow(-1), g.inverse());
    EXPECT_EQ(g.pow(3), g * g * g);
    EXPECT_TRUE(g.pow(0).is_one());
}

TEST(GtTest, EncodingRoundTripsIncludingOne){
    const auto x = Gt::random();

    EXPECT_EQ(Gt::from_bytes(x.to_bytes()), x);
    EXPECT_TRUE(Gt::from_bytes(Gt().to_bytes()).is_one());
}

TEST(GtTest, DecodingRejectsInvalidEncodings){
    const auto valid = Gt::generator().to_bytes();
    ASSERT_EQ(Gt::from_bytes(valid), Gt::generator());

    EXPECT_THROW((void)Gt::from_bytes(Bytes(valid.size(), 0xFF)), DecodeError);
    EXPECT_THROW((void)Gt::from_bytes(ByteView(valid).first(valid.size() - 1)), DecodeError);
}

TEST(GtTest, DecodingAgreesWithBlstInGroup){
    auto outside = Gt::generator().to_bytes();
    outside.back() ^= 0x01;
    const std::vector<Candidate> candidates{
        {"generator", Gt::generator().to_bytes(), true},
        {"one", Gt().to_bytes(), true},
        {"random element", Gt::random().to_bytes(), true},
        {"perturbed generator", outside, false},
        {"cyclotomic element of the wrong order", blst_bytes(easy_part(blst_from_bytes(outside))), false},
        {"zero", Bytes(Gt::byte_size, 0), false},
    };
    for (const auto& candidate : candidates){
        const auto f = blst_from_bytes(candidate.bytes);
        EXPECT_EQ(blst_fp12_in_group(&f), candidate.in_group) << candidate.name;
        EXPECT_EQ(decodes(candidate.bytes), candidate.in_group) << candidate.name;
    }
}
