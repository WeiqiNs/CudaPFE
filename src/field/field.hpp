#ifndef CUDAPFE_FIELD_FIELD_HPP
#define CUDAPFE_FIELD_FIELD_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <ff/bls12-381.hpp>
#include "field/constants.hpp"
#include "support/hd.hpp"
#include "support/words.hpp"

namespace cudapfe::detail{
    struct FpParams{
        using Base = bls12_381::fp_t;
        static constexpr std::size_t words = 6;
        static constexpr Words<6> modulus = kP;
        static constexpr Words<6> inverse_exponent = kFpInverseExponent;
    };

    struct FrParams{
        using Base = bls12_381::fr_t;
        static constexpr std::size_t words = 4;
        static constexpr Words<4> modulus = kR;
        static constexpr Words<4> inverse_exponent = kFrInverseExponent;
    };

    template <class T, std::size_t M>
    CUDAPFE_HD T pow(const T& base, const Words<M>& exponent){
        auto result = T::one();
#pragma unroll 1
        for (auto i = bit_length(exponent); i-- > 0;){
            result = result.square();
            if (bit(exponent, i)) result = result * base;
        }
        return result;
    }

    template <class Params>
    struct alignas(Params::words == 6 ? 16 : 32) Field{
        static constexpr std::size_t N = Params::words;
        static_assert(sizeof(typename Params::Base) == N * sizeof(Word));

        typename Params::Base value;

        [[nodiscard]] CUDAPFE_HD static Field zero(){ return from_montgomery(Words<N>{}); }

        [[nodiscard]] CUDAPFE_HD static Field one(){ return Field{Params::Base::one()}; }

        [[nodiscard]] CUDAPFE_HD static Field from_montgomery(const Words<N>& words){
            Field field;
#ifdef __CUDA_ARCH__
            for (std::size_t i = 0; i < N; ++i){
                field.value[2 * i] = static_cast<std::uint32_t>(words[i]);
                field.value[2 * i + 1] = static_cast<std::uint32_t>(words[i] >> 32);
            }
#else
            field.value = typename Params::Base(words.data());
#endif
            return field;
        }

        [[nodiscard]] CUDAPFE_HD static Field from_canonical(const Words<N>& words){
            auto field = from_montgomery(words);
            field.value.to();
            return field;
        }

        [[nodiscard]] CUDAPFE_HD Words<N> montgomery() const{
            Words<N> words;
#ifdef __CUDA_ARCH__
            for (std::size_t i = 0; i < N; ++i) words[i] = Word{value[2 * i]} | Word{value[2 * i + 1]} << 32;
#else
            value.store(words.data());
#endif
            return words;
        }

        [[nodiscard]] CUDAPFE_HD Words<N> canonical() const{
            auto field = *this;
            field.value.from();
            return field.montgomery();
        }

        [[nodiscard]] CUDAPFE_HD bool is_zero() const{ return value.is_zero(); }

        [[nodiscard]] CUDAPFE_HD Field square() const{ return Field{sqr(value)}; }

        [[nodiscard]] CUDAPFE_HD Field inverse() const{
#ifdef __CUDA_ARCH__
            constexpr auto exponent = Params::inverse_exponent;
            return pow(exponent);
#else
            return Field{value.reciprocal()};
#endif
        }

        template <std::size_t M>
        [[nodiscard]] CUDAPFE_HD Field pow(const Words<M>& exponent) const{ return detail::pow(*this, exponent); }

        friend CUDAPFE_HD Field operator+(const Field& x, const Field& y){ return Field{x.value + y.value}; }

        friend CUDAPFE_HD Field operator-(const Field& x, const Field& y){ return Field{x.value - y.value}; }

        friend CUDAPFE_HD Field operator*(const Field& x, const Field& y){ return Field{x.value * y.value}; }

        friend CUDAPFE_HD Field operator-(const Field& x){
            auto negated = x;
            negated.value.cneg(true);
            return negated;
        }

        friend CUDAPFE_HD bool operator==(const Field& x, const Field& y){
            const auto left = x.montgomery();
            const auto right = y.montgomery();
            Word difference = 0;
            for (std::size_t i = 0; i < N; ++i) difference |= left[i] ^ right[i];
            return difference == 0;
        }
    };

    using Fp = Field<FpParams>;
    using Fr = Field<FrParams>;

    static_assert(sizeof(Fp) == 48 && alignof(Fp) == 16 && std::is_trivially_copyable_v<Fp>);
    static_assert(sizeof(Fr) == 32 && alignof(Fr) == 32 && std::is_trivially_copyable_v<Fr>);
}

#endif
