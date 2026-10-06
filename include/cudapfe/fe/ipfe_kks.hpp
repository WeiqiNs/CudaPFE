#ifndef CUDAPFE_FE_IPFE_KKS_HPP
#define CUDAPFE_FE_IPFE_KKS_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "ipfe.hpp"

namespace cudapfe::IPFE::KKS{
    template <Engine E>
    struct Msk{
        Zp eta;
        Zp eta_bar;
        Vector s;
        Vector t;
        Vector u;
        Vector v;
        Vector h;
        Vector h_hat;
        Vector h_bar;
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

    namespace detail{
        template <Engine E>
        [[nodiscard]] Vector ciphertext_half(const Msk<E>& msk, const Vector& h, const Vector& m){
            const auto r = Zp::random();
            const auto ct1 = concat({Vector{r, msk.eta * r}, m + h * r});
            return concat({Vector{-inner(msk.u, ct1), -inner(msk.v, ct1)}, ct1});
        }

        template <Engine E>
        [[nodiscard]] Vector key_half(const Msk<E>& msk, const Vector& key){
            const auto r = Zp::random();
            return concat({Vector{r, msk.eta_bar * r}, key + msk.h_bar * r});
        }
    }

    template <Engine E>
    [[nodiscard]] Msk<E> setup(const std::size_t size){
        const auto eta = Zp::random();
        const auto eta_bar = Zp::random();
        auto s = random_vector(size);
        auto t = random_vector(size);
        auto u = random_vector(size + 2);
        auto v = random_vector(size + 2);
        auto h = s + t * eta;
        auto h_hat = random_vector(size) + random_vector(size) * eta;
        auto h_bar = u + v * eta_bar;
        return {
            eta, eta_bar, std::move(s), std::move(t), std::move(u), std::move(v), std::move(h), std::move(h_hat),
            std::move(h_bar)
        };
    }

    [[nodiscard]] inline Gt base(){
        return Gt::generator();
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntMatrix& functions){
        Vector exponents;
        for (const auto& function : functions){
            const auto f = to_vector(function);
            const auto key = concat({Vector{-inner(msk.s, f), -inner(msk.t, f)}, f});
            IPFE::detail::append(
                exponents, concat({detail::key_half(msk, key), detail::key_half(msk, Vector(key.size()))})
            );
        }
        return {functions.size(), IPFE::detail::lift<G2, E>(exponents)};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntVec& function){
        return keygen(msk, IntMatrix{function});
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntMatrix& messages){
        Vector exponents;
        for (const auto& message : messages){
            const auto m = to_vector(message);
            IPFE::detail::append(
                exponents, concat({detail::ciphertext_half(msk, msk.h, m), detail::ciphertext_half(msk, msk.h_hat, m)})
            );
        }
        return {messages.size(), IPFE::detail::lift<G1, E>(exponents)};
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
        const auto batch = IPFE::detail::broadcast({.keys = sk.count, .ciphertexts = ct.count});
        return table.find(IPFE::detail::pair_entries(ct.vec, sk.vec, batch));
    }
}

#endif
