#ifndef CUDAPFE_SUPPORT_WORDS_HPP
#define CUDAPFE_SUPPORT_WORDS_HPP

#include <cstddef>
#include <cstdint>
#include <cudapfe/core.hpp>
#include "support/hd.hpp"

namespace cudapfe::detail{
    template <std::size_t N>
    constexpr CUDAPFE_HD Words<N> add_word(Words<N> words, Word carry){
        for (auto& word : words){
            word += carry;
            carry = word < carry;
        }
        return words;
    }

    template <std::size_t N>
    constexpr CUDAPFE_HD Words<N> sub_word(Words<N> words, Word borrow){
        for (auto& word : words){
            const auto before = word;
            word -= borrow;
            borrow = word > before;
        }
        return words;
    }

    template <std::size_t N>
    constexpr CUDAPFE_HD Words<N> shift_right(const Words<N>& words, const unsigned shift){
        Words<N> shifted{};
        const std::size_t skip = shift / 64;
        const unsigned bits = shift % 64;
        for (std::size_t i = 0; i + skip < N; ++i){
            shifted[i] = words[i + skip] >> bits;
            if (bits != 0 && i + skip + 1 < N) shifted[i] |= words[i + skip + 1] << (64 - bits);
        }
        return shifted;
    }

    template <std::size_t N>
    constexpr CUDAPFE_HD bool less(const Words<N>& x, const Words<N>& y){
        for (std::size_t i = N; i-- > 0;){
            if (x[i] != y[i]) return x[i] < y[i];
        }
        return false;
    }

    template <std::size_t N>
    constexpr CUDAPFE_HD std::size_t bit_length(const Words<N>& words){
        for (std::size_t i = N; i-- > 0;){
            if (words[i] == 0) continue;
            std::size_t length = 64 * i;
            for (auto top = words[i]; top != 0; top >>= 1) ++length;
            return length;
        }
        return 0;
    }

    template <std::size_t N>
    constexpr CUDAPFE_HD bool bit(const Words<N>& words, const std::size_t index){
        return (words[index / 64] >> (index % 64)) & 1;
    }

    template <std::size_t N, std::size_t D>
    struct Division{
        Words<N> quotient;
        Words<D> remainder;
    };

    template <std::size_t N, std::size_t D>
    constexpr CUDAPFE_HD Division<N, D> divide(const Words<N>& dividend, const Words<D>& divisor){
        Division<N, D> result{};
        auto& remainder = result.remainder;
#pragma unroll 1
        for (auto i = 64 * N; i-- > 0;){
            const auto carry = remainder[D - 1] >> 63;
            for (std::size_t j = D; j-- > 1;) remainder[j] = remainder[j] << 1 | remainder[j - 1] >> 63;
            remainder[0] = remainder[0] << 1 | Word{bit(dividend, i)};
            if (carry == 0 && less(remainder, divisor)) continue;
            Word borrow = 0;
            for (std::size_t j = 0; j < D; ++j){
                const auto difference = remainder[j] - divisor[j];
                const Word next_borrow = remainder[j] < divisor[j] || difference < borrow;
                remainder[j] = difference - borrow;
                borrow = next_borrow;
            }
            result.quotient[i / 64] |= Word{1} << (i % 64);
        }
        return result;
    }

    template <std::size_t N>
    void append_big_endian(Bytes& out, const Words<N>& words){
        for (auto word = words.rbegin(); word != words.rend(); ++word){
            for (int shift = 56; shift >= 0; shift -= 8) out.push_back(static_cast<std::uint8_t>(*word >> shift));
        }
    }

    template <std::size_t N>
    [[nodiscard]] Words<N> read_big_endian(const ByteView bytes){
        Words<N> words{};
        for (std::size_t i = 0; i < 8 * N; ++i){
            auto& word = words[N - 1 - i / 8];
            word = word << 8 | bytes[i];
        }
        return words;
    }
}

#endif
