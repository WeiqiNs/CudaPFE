#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <support/oracle.hpp>
#include <support/samples.hpp>
#include "curve/curve.hpp"
#include "curve/encoding.hpp"
#include "field/constants.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"

using namespace cufe;
using namespace cufe::detail;
using namespace cufe::test;

namespace{
    constexpr std::size_t kPoints = 100;
    constexpr std::size_t kRoots = 100;

    struct Malformed{
        std::string_view defect;
        Bytes bytes;
        bool blst_has_this_view;
    };

    template <class P>
    Bytes blst_bytes(void (*write)(byte*, const P*), const P& p, const std::size_t size){
        Bytes out(size);
        write(out.data(), &p);
        return out;
    }

    template <class F>
    bool blst_rejects(const Bytes& bytes){
        using O = CurveOracle<F>;
        Blst<Affine<F>> point;
        const auto error = bytes.size() == encoded_size<F>(Encoding::compressed)
            ? O::uncompress(&point, bytes.data())
            : O::deserialize(&point, bytes.data());
        return error != BLST_SUCCESS || !O::in_group(&point);
    }

    Bytes modulus_bytes(){
        Bytes out;
        append_big_endian(out, kP);
        return out;
    }

    template <class F>
    Bytes first_x_off_the_curve(){
        for (std::uint8_t x = 1;; ++x){
            auto bytes = small_x_compressed<F>(x);
            Blst<Affine<F>> point;
            if (CurveOracle<F>::uncompress(&point, bytes.data()) == BLST_POINT_NOT_ON_CURVE) return bytes;
        }
    }

    template <class F>
    std::vector<Malformed> malformed_encodings(){
        const auto generator = CurveParams<F>::generator();
        const auto compressed = encode(generator, Encoding::compressed);
        const auto uncompressed = encode(generator, Encoding::uncompressed);
        const auto infinity = encode(Affine<F>::identity(), Encoding::compressed);
        const auto modulus = modulus_bytes();

        auto truncated = compressed;
        truncated.pop_back();
        auto unflagged = compressed;
        unflagged.front() &= ~kCompressedFlag;
        auto flagged = uncompressed;
        flagged.front() |= kCompressedFlag;
        auto infinity_with_byte = infinity;
        infinity_with_byte.back() = 1;
        auto infinity_with_sign = infinity;
        infinity_with_sign.front() |= kSignFlag;
        auto signed_uncompressed = uncompressed;
        signed_uncompressed.front() |= kSignFlag;
        auto x_is_p = Bytes(compressed.size(), 0);
        std::copy(modulus.begin(), modulus.end(), x_is_p.begin());
        x_is_p.front() |= kCompressedFlag;
        auto y_is_p = uncompressed;
        std::copy(modulus.begin(), modulus.end(), y_is_p.begin() + compressed.size());
        auto off_curve = uncompressed;
        off_curve.back() ^= 1;

        return {
            {"length is neither point size", truncated, false},
            {"compressed length without the compression flag", unflagged, true},
            {"uncompressed length with the compression flag", flagged, false},
            {"infinity flag with a nonzero byte", infinity_with_byte, true},
            {"infinity flag with the sign flag", infinity_with_sign, true},
            {"sign flag on an uncompressed point", signed_uncompressed, true},
            {"x coordinate equal to p", x_is_p, true},
            {"y coordinate equal to p", y_is_p, true},
            {"compressed x with no curve point", first_x_off_the_curve<F>(), true},
            {"uncompressed point off the curve", off_curve, true},
            {"point outside the subgroup", encode(first_point_outside_subgroup<F>(), Encoding::compressed), true},
        };
    }
}

template <class F>
class EncodingTest : public ::testing::Test{};

TYPED_TEST_SUITE(EncodingTest, Sides, SideNames);

TYPED_TEST(EncodingTest, MatchesBlst){
    using F = TypeParam;
    using O = CurveOracle<F>;
    for (const auto& jacobian : random_points<F>(kPoints)){
        const auto p = to_affine(jacobian);
        const auto compressed = encode(p, Encoding::compressed);
        const auto uncompressed = encode(p, Encoding::uncompressed);
        EXPECT_EQ(compressed, blst_bytes(O::compress, to_blst(p), encoded_size<F>(Encoding::compressed)));
        EXPECT_EQ(uncompressed, blst_bytes(O::serialize, to_blst(p), encoded_size<F>(Encoding::uncompressed)));
        EXPECT_EQ(decode<F>(compressed), p);
        EXPECT_EQ(decode<F>(uncompressed), p);
    }

    Bytes compressed_identity(encoded_size<F>(Encoding::compressed), 0);
    compressed_identity.front() = 0xc0;
    Bytes uncompressed_identity(encoded_size<F>(Encoding::uncompressed), 0);
    uncompressed_identity.front() = 0x40;
    EXPECT_EQ(encode(Affine<F>::identity(), Encoding::compressed), compressed_identity);
    EXPECT_EQ(encode(Affine<F>::identity(), Encoding::uncompressed), uncompressed_identity);
    EXPECT_TRUE(decode<F>(compressed_identity).is_identity());
    EXPECT_TRUE(decode<F>(uncompressed_identity).is_identity());
}

TYPED_TEST(EncodingTest, RejectsMalformedEncodings){
    using F = TypeParam;
    for (const auto& row : malformed_encodings<F>()){
        EXPECT_THROW((void)decode<F>(row.bytes), DecodeError) << row.defect;
        if (row.blst_has_this_view) EXPECT_TRUE(blst_rejects<F>(row.bytes)) << row.defect;
    }
}

TEST(EncodingTest, SquareRootsSquareBack){
    const Fp2 one_plus_u{Fp::one(), Fp::one()};
    for (const auto& x : field_samples<FpParams>(kRoots)){
        const auto square = x.square();
        const auto root = sqrt(square);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->square(), square);
        if (x.is_zero()) continue;
        const auto nonresidue = to_blst(-square);
        EXPECT_FALSE(sqrt(-square));
        EXPECT_FALSE(blst_fp_is_square(&nonresidue));
    }
    for (const auto& x : tower_samples<Fp2>(kRoots)){
        const auto square = x.square();
        const auto root = sqrt(square);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->square(), square);
        const auto nonresidue = to_blst(square * one_plus_u);
        EXPECT_FALSE(sqrt(square * one_plus_u));
        EXPECT_FALSE(blst_fp2_is_square(&nonresidue));
    }
}

TEST(EncodingTest, SignFlagMarksTheUpperHalfOfTheField){
    const auto half = Fp::from_canonical(kFpHalfExponent);
    const auto above = half + Fp::one();
    EXPECT_FALSE(is_lexicographically_largest(half));
    EXPECT_TRUE(is_lexicographically_largest(above));
    EXPECT_TRUE(is_lexicographically_largest(Fp2{above, Fp::zero()}));
    EXPECT_FALSE(is_lexicographically_largest(Fp2{above, half}));
    EXPECT_TRUE(is_lexicographically_largest(Fp2{half, above}));
}
