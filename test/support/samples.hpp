#ifndef CUFE_TEST_SAMPLES_HPP
#define CUFE_TEST_SAMPLES_HPP

#include <bit>
#include <cstddef>
#include <random>
#include <vector>
#include "field/field.hpp"

namespace cufe::test{
    template <class Params>
    std::vector<detail::Words<Params::words>> canonical_samples(const std::size_t random_count){
        using Words = detail::Words<Params::words>;
        constexpr auto modulus = Params::modulus;
        constexpr auto top_mask = (detail::Word{1} << std::bit_width(modulus.back())) - 1;
        std::mt19937_64 rng(random_count);
        std::vector<Words> samples{Words{}, Words{1}, detail::sub_word(modulus, 1)};
        while (samples.size() < random_count + 3){
            Words words;
            for (auto& word : words) word = rng();
            words.back() &= top_mask;
            if (detail::less(words, modulus)) samples.push_back(words);
        }
        return samples;
    }

    template <class Params>
    std::vector<detail::Field<Params>> field_samples(const std::size_t random_count){
        std::vector<detail::Field<Params>> samples;
        for (const auto& words : canonical_samples<Params>(random_count)){
            samples.push_back(detail::Field<Params>::from_canonical(words));
        }
        return samples;
    }
}

#endif
