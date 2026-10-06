#ifndef CUDAPFE_FE_IPFE_BJK_HPP
#define CUDAPFE_FE_IPFE_BJK_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "ipfe.hpp"

namespace cudapfe::IPFE::BJK{
    template <Engine E>
    struct Msk{
        Matrix<E> b;
        Matrix<E> bi;
        Matrix<E> d;
        Matrix<E> di;
    };

    template <Engine E>
    struct Sk{
        std::size_t count;
        Vec<G2, E> r;
        Vec<G2, E> vec;
    };

    template <Engine E>
    struct Ct{
        std::size_t count;
        Vec<G1, E> r;
        Vec<G1, E> vec;
    };

    template <Engine E>
    struct PreparedSk{
        std::size_t count;
        G2Lines<E> r;
        G2Lines<E> vec;
    };

    template <Engine E>
    [[nodiscard]] Msk<E> setup(const std::size_t size){
        detail::require_setup_memory<E>(2 * size + 4, "IPFE::BJK");
        auto b = Matrix<E>::random({2 * size + 4, 2 * size + 4});
        auto bi = b.inverse().transpose();
        auto d = Matrix<E>::random({2, 2});
        auto di = d.inverse().transpose();
        return {std::move(b), std::move(bi), std::move(d), std::move(di)};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntMatrix& functions){
        std::vector<Vector> randomness;
        std::vector<Vector> encoded;
        for (const auto& function : functions){
            const auto f = to_vector(function);
            const auto beta = Zp::random();
            const auto beta_t = Zp::random();
            randomness.push_back({beta, beta_t});
            encoded.push_back(concat({f * beta, f * beta_t, Vector{{}, beta, {}, beta_t}}));
        }
        return {functions.size(), detail::lift<G2>(randomness, msk.d), detail::lift<G2>(encoded, msk.b)};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntVec& function){
        return keygen(msk, IntMatrix{function});
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntMatrix& messages){
        std::vector<Vector> randomness;
        std::vector<Vector> encoded;
        for (const auto& message : messages){
            const auto m = to_vector(message);
            const auto alpha = Zp::random();
            const auto alpha_t = Zp::random();
            randomness.push_back({alpha, alpha_t});
            encoded.push_back(concat({m * alpha, m * alpha_t, Vector{alpha, {}, alpha_t, {}}}));
        }
        return {messages.size(), detail::lift<G1>(randomness, msk.di), detail::lift<G1>(encoded, msk.bi)};
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntVec& message){
        return enc(msk, IntMatrix{message});
    }

    template <Engine E>
    [[nodiscard]] PreparedSk<E> prepare(const Sk<E>& sk){
        return {sk.count, cudapfe::prepare(sk.r), cudapfe::prepare(sk.vec)};
    }

    template <Engine E, class Key> requires std::same_as<Key, Sk<E>> || std::same_as<Key, PreparedSk<E>>
    [[nodiscard]] std::vector<std::optional<std::int64_t>> dec(const Key& sk, const Ct<E>& ct, const Range& range){
        const auto batch = detail::broadcast({.keys = sk.count, .ciphertexts = ct.count});
        const DlogTables<E> tables(detail::pair_entries(ct.r, sk.r, batch), range);
        return tables.find(detail::pair_entries(ct.vec, sk.vec, batch));
    }
}

#endif
