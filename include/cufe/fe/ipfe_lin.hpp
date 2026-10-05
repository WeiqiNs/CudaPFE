#ifndef CUFE_FE_IPFE_LIN_HPP
#define CUFE_FE_IPFE_LIN_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include "ipfe.hpp"

namespace cufe::IPFE::LIN{
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

    [[nodiscard]] inline Gt base(){
        return Gt::generator();
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
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntVec& function){
        return keygen(msk, IntMatrix{function});
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
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntVec& message){
        return enc(msk, IntMatrix{message});
    }

    template <Engine E>
    [[nodiscard]] PreparedSk<E> prepare(const Sk<E>& sk){
        return {sk.count, cufe::prepare(sk.vec)};
    }

    template <Engine E, class Key> requires std::same_as<Key, Sk<E>> || std::same_as<Key, PreparedSk<E>>
    [[nodiscard]] std::vector<std::optional<std::int64_t>> dec(
        const DlogTable<E>& table, const Key& sk, const Ct<E>& ct
    ){
        const auto batch = detail::broadcast({.keys = sk.count, .ciphertexts = ct.count});
        return table.find(detail::pair_entries(ct.vec, sk.vec, batch));
    }
}

#endif
