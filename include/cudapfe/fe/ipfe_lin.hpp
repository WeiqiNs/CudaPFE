#ifndef CUDAPFE_FE_IPFE_LIN_HPP
#define CUDAPFE_FE_IPFE_LIN_HPP

#include <cstddef>
#include "ipfe.hpp"

namespace cudapfe::IPFE::LIN{
    using IPFE::base, IPFE::dec, IPFE::enc, IPFE::keygen;

    template <Engine E>
    struct Msk{
        Vector s1;
        Vector s2;
    };

    template <Engine E>
    struct Sk{
        std::size_t count;
        Vec<G2, E> vec;
    };

    template <Engine E>
    struct Ct{
        std::size_t count;
        Vec<G1, E> vec;
    };

    template <Engine E>
    struct PreparedSk{
        std::size_t count;
        G2Lines<E> vec;
    };

    template <Engine E>
    [[nodiscard]] Msk<E> setup(const std::size_t size){
        return {random_vector(2 * size), random_vector(2 * size + 1)};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntMatrix& functions){
        Vector exponents;
        for (const auto& function : functions){
            const auto f = concat({to_vector(function), Vector(function.size())});
            const auto key = concat({Vector{inner(f, msk.s1)}, f});
            const auto r = Zp::random();
            detail::append(exponents, concat({Vector{-r}, msk.s2 * r + key}));
        }
        return {functions.size(), detail::lift<G2, E>(exponents)};
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntMatrix& messages){
        Vector exponents;
        for (const auto& message : messages){
            const auto m = concat({to_vector(message), Vector(message.size())});
            const auto r = Zp::random();
            const auto ct = concat({Vector{-r}, msk.s1 * r + m});
            detail::append(exponents, concat({Vector{inner(msk.s2, ct)}, ct}));
        }
        return {messages.size(), detail::lift<G1, E>(exponents)};
    }

    template <Engine E>
    [[nodiscard]] PreparedSk<E> prepare(const Sk<E>& sk){
        return {sk.count, cudapfe::prepare(sk.vec)};
    }
}

#endif
