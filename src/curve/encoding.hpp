#ifndef CUDAPFE_CURVE_ENCODING_HPP
#define CUDAPFE_CURVE_ENCODING_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <cudapfe/core.hpp>
#include <cudapfe/errors.hpp>
#include "curve/curve.hpp"
#include "field/constants.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/words.hpp"

namespace cudapfe::detail{
    inline constexpr std::uint8_t kCompressedFlag = 0x80;
    inline constexpr std::uint8_t kInfinityFlag = 0x40;
    inline constexpr std::uint8_t kSignFlag = 0x20;
    inline constexpr std::uint8_t kFlagMask = kCompressedFlag | kInfinityFlag | kSignFlag;
    inline constexpr std::size_t kFpBytes = 48;

    template <class F>
    inline constexpr std::string_view kGroupName = std::same_as<F, Fp> ? "G1" : "G2";

    [[nodiscard]] inline std::optional<Fp> sqrt(const Fp& x){
        constexpr auto exponent = kFpSqrtExponent;
        const auto root = x.pow(exponent);
        if (root.square() != x) return std::nullopt;
        return root;
    }

    [[nodiscard]] inline std::optional<Fp2> sqrt(const Fp2& x){
        constexpr auto exponent = kFp2SqrtExponent;
        constexpr auto half = kFpHalfExponent;
        const auto minus_one = -Fp2::one();
        const auto a = pow(x, exponent);
        const auto alpha = a.square() * x;
        if (conjugate(alpha) * alpha == minus_one) return std::nullopt;
        const auto root = a * x;
        if (alpha == minus_one) return Fp2{-root.c1, root.c0};
        return pow(Fp2::one() + alpha, half) * root;
    }

    [[nodiscard]] inline bool is_lexicographically_largest(const Fp& x){
        constexpr auto half = kFpHalfExponent;
        return less(half, x.canonical());
    }

    [[nodiscard]] inline bool is_lexicographically_largest(const Fp2& x){
        return is_lexicographically_largest(x.c1.is_zero() ? x.c0 : x.c1);
    }

    template <class F>
    [[nodiscard]] constexpr std::size_t encoded_size(const Encoding encoding){
        constexpr std::size_t coordinate = std::same_as<F, Fp> ? kFpBytes : 2 * kFpBytes;
        return encoding == Encoding::compressed ? coordinate : 2 * coordinate;
    }

    inline void append(Bytes& out, const Fp& x){
        append_big_endian(out, x.canonical());
    }

    inline void append(Bytes& out, const Fp2& x){
        append(out, x.c1);
        append(out, x.c0);
    }

    [[nodiscard]] inline Fp read_fp(const ByteView bytes){
        const auto words = read_big_endian<6>(bytes);
        constexpr auto modulus = kP;
        if (!less(words, modulus)) throw DecodeError("an Fp encoding is not below the field modulus");
        return Fp::from_canonical(words);
    }

    template <class F>
    [[nodiscard]] F read_coordinate(const ByteView bytes){
        if constexpr (std::same_as<F, Fp>) return read_fp(bytes);
        else return {read_fp(bytes.subspan(kFpBytes)), read_fp(bytes.first(kFpBytes))};
    }

    template <class F>
    [[nodiscard]] Bytes encode(const Affine<F>& p, const Encoding encoding){
        Bytes out;
        out.reserve(encoded_size<F>(encoding));
        if (p.is_identity()){
            out.resize(encoded_size<F>(encoding), 0);
            out.front() = encoding == Encoding::compressed ? kCompressedFlag | kInfinityFlag : kInfinityFlag;
            return out;
        }
        append(out, p.x);
        if (encoding == Encoding::uncompressed) append(out, p.y);
        else out.front() |= kCompressedFlag | (is_lexicographically_largest(p.y) ? kSignFlag : 0);
        return out;
    }

    template <class F>
    [[nodiscard]] Affine<F> lift(const F& x, const bool largest){
        const auto y = sqrt(x.square() * x + CurveParams<F>::b());
        if (!y) throw DecodeError(std::string(kGroupName<F>) + " compressed x is not the abscissa of a curve point");
        return {x, is_lexicographically_largest(*y) == largest ? *y : -*y};
    }

    template <class F>
    [[nodiscard]] Affine<F> decode(const ByteView bytes){
        const std::string name(kGroupName<F>);
        constexpr auto compressed_size = encoded_size<F>(Encoding::compressed);
        const bool compressed = bytes.size() == compressed_size;
        if (!compressed && bytes.size() != encoded_size<F>(Encoding::uncompressed)){
            throw DecodeError(name + " encoding of " + std::to_string(bytes.size()) + " bytes has neither point size");
        }
        const auto flags = bytes.front() & kFlagMask;
        if (((flags & kCompressedFlag) != 0) != compressed){
            throw DecodeError(name + " compression flag disagrees with the encoding length");
        }
        if (flags & kInfinityFlag){
            const auto zero = [](const std::uint8_t byte){ return byte == 0; };
            const bool only_flags = (bytes.front() & ~(kCompressedFlag | kInfinityFlag)) == 0;
            if (!only_flags || !std::all_of(bytes.begin() + 1, bytes.end(), zero)){
                throw DecodeError(name + " infinity encoding has other bits set");
            }
            return Affine<F>::identity();
        }
        if (!compressed && (flags & kSignFlag)) throw DecodeError(name + " uncompressed encoding has the sign flag");

        Bytes x_bytes(bytes.begin(), bytes.begin() + compressed_size);
        x_bytes.front() &= ~kFlagMask;
        const auto x = read_coordinate<F>(x_bytes);
        const auto point = compressed
            ? lift(x, flags & kSignFlag)
            : Affine<F>{x, read_coordinate<F>(bytes.subspan(compressed_size))};
        if (!compressed && !on_curve(point)) throw DecodeError(name + " uncompressed encoding is not a curve point");
        if (!in_subgroup(point)) throw DecodeError(name + " encoding is not a point of the prime-order subgroup");
        return point;
    }
}

#endif
