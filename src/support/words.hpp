#ifndef CUFE_SUPPORT_WORDS_HPP
#define CUFE_SUPPORT_WORDS_HPP

#include <cstddef>
#include "support/hd.hpp"

namespace cufe::detail{
    template <std::size_t N>
    constexpr CUFE_HD Words<N> add_word(Words<N> words, Word carry){
        for (auto& word : words){
            word += carry;
            carry = word < carry;
        }
        return words;
    }

    template <std::size_t N>
    constexpr CUFE_HD Words<N> sub_word(Words<N> words, Word borrow){
        for (auto& word : words){
            const auto before = word;
            word -= borrow;
            borrow = word > before;
        }
        return words;
    }

    template <std::size_t N>
    constexpr CUFE_HD Words<N> shift_right(const Words<N>& words, const unsigned shift){
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
    constexpr CUFE_HD bool less(const Words<N>& x, const Words<N>& y){
        for (std::size_t i = N; i-- > 0;){
            if (x[i] != y[i]) return x[i] < y[i];
        }
        return false;
    }

    template <std::size_t N>
    constexpr CUFE_HD std::size_t bit_length(const Words<N>& words){
        for (std::size_t i = N; i-- > 0;){
            if (words[i] == 0) continue;
            std::size_t length = 64 * i;
            for (auto top = words[i]; top != 0; top >>= 1) ++length;
            return length;
        }
        return 0;
    }

    template <std::size_t N>
    constexpr CUFE_HD bool bit(const Words<N>& words, const std::size_t index){
        return (words[index / 64] >> (index % 64)) & 1;
    }
}

#endif
