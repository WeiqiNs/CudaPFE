#ifndef CUDAPFE_FE_IPFE_TAO_HPP
#define CUDAPFE_FE_IPFE_TAO_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "ipfe.hpp"

namespace cudapfe::IPFE::TAO{
    template <Engine E>
    struct Msk{
        Matrix<E> b;
        Matrix<E> bi;
        Gt base;
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
        detail::require_setup_memory<E>(2 * size + 5, "IPFE::TAO");
        const auto r = Zp::random();
        auto b = Matrix<E>::random({2 * size + 5, 2 * size + 5});
        auto bi = (b.inverse() * r).transpose();
        return {std::move(b), std::move(bi), Gt::generator().pow(r)};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntMatrix& functions){
        std::vector<Vector> encoded;
        for (const auto& function : functions){
            const auto f = to_vector(function);
            encoded.push_back(concat({f, Vector(f.size() + 2), Vector{Zp::random(), Zp::random(), {}}}));
        }
        return {functions.size(), detail::lift<G2>(encoded, msk.b)};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntVec& function){
        return keygen(msk, IntMatrix{function});
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntMatrix& messages){
        std::vector<Vector> encoded;
        for (const auto& message : messages){
            const auto m = to_vector(message);
            encoded.push_back(concat({m, Vector(m.size()), Vector{Zp::random(), Zp::random(), {}, {}, {}}}));
        }
        return {messages.size(), detail::lift<G1>(encoded, msk.bi)};
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntVec& message){
        return enc(msk, IntMatrix{message});
    }

    template <Engine E>
    [[nodiscard]] PreparedSk<E> prepare(const Sk<E>& sk){
        return {sk.count, cudapfe::prepare(sk.vec)};
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
