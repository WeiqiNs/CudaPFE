#ifndef CUDAPFE_FE_QFE_BCFG_HPP
#define CUDAPFE_FE_QFE_BCFG_HPP

#include <cstddef>
#include <utility>
#include <vector>
#include "qfe.hpp"

namespace cudapfe::QFE::BCFG{
    using IPFE::base, QFE::enc, QFE::keygen;

    template <Engine E>
    struct Msk{
        Zp w;
        Vector a;
        Vector b;
    };

    template <Engine E>
    struct Pk{
        Vec<G1, E> a;
        Vec<G2, E> b;
        G2 w;
    };

    template <Engine E>
    struct Keys{
        Pk<E> pk;
        Msk<E> msk;
    };

    template <Engine E>
    struct Sk{
        Vec<Zp, E> f;
        Vec<G1, E> s1;
        Vec<G1, E> s2;
    };

    template <Engine E>
    struct Ct{
        Vec<G1, E> c;
        Vec<G1, E> c_hat;
        Vec<G2, E> d;
        Vec<G2, E> d_hat;
        Vec<G2, E> e;
        Vec<G2, E> e_hat;
    };

    template <Engine E>
    [[nodiscard]] Keys<E> setup(const std::size_t size){
        const auto w = Zp::random();
        auto a = random_vector(size);
        auto b = random_vector(size);
        return {
            {detail::lift<G1, E>(a), detail::lift<G2, E>(b), G2::mul_generator(w)},
            {w, std::move(a), std::move(b)}
        };
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const std::vector<IntMatrix>& functions){
        Vector f;
        Vector s1;
        Vector s2;
        for (const auto& function : functions){
            const auto rows = detail::rows(function, msk.a.size(), "QFE::BCFG::keygen");
            for (const auto& row : rows) detail::append(f, row);
            const auto gamma = Zp::random();
            s1.push_back(detail::quadratic_form(msk.a, rows, msk.b) + gamma * msk.w);
            s2.push_back(gamma);
        }
        return {Vec<Zp, E>::upload(f), detail::lift<G1, E>(s1), detail::lift<G1, E>(s2)};
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Pk<E>& pk, const IntMatrix& lefts, const IntMatrix& rights){
        detail::require_pairs(lefts, rights, "QFE::BCFG::enc");
        detail::Mask c;
        detail::Mask c_hat;
        detail::Mask d;
        detail::Mask d_hat;
        Vector blind;
        for (std::size_t i = 0; i < lefts.size(); ++i){
            const auto x = detail::row(lefts[i], pk.a.size(), "QFE::BCFG::enc");
            const auto y = detail::row(rights[i], pk.b.size(), "QFE::BCFG::enc");
            const auto r = Zp::random();
            const auto s = Zp::random();
            const auto t = Zp::random();
            const auto z = Zp::random();
            c.add(r, x);
            c_hat.add(t, x * s);
            d.add(s, y);
            d_hat.add(z, y * r);
            blind.push_back(r * s - z - t);
        }
        return {
            detail::masked(pk.a, c),
            detail::masked(pk.a, c_hat),
            detail::masked(pk.b, d),
            detail::masked(pk.b, d_hat),
            detail::lift<G2, E>(blind),
            detail::scaled_copies(Vec<G2, E>::upload(std::vector{pk.w}), blind)
        };
    }

    template <Engine E>
    [[nodiscard]] Results dec(
        const DlogTable<E>& table, const Pk<E>& pk, const Sk<E>& sk, const Ct<E>& ct
    ){
        const auto batch = detail::broadcast({.keys = sk.s1.size(), .ciphertexts = ct.e.size()});
        const auto n = pk.a.size();
        const auto ps = concat<G1, E>(batch.segments, {
            {detail::bilinear(ct.c, sk.f, n, batch), n},
            {msm(-pk.a, sk.f, MsmShape{sk.s1.size(), {n, n}, Spread::shared}), n, batch.keys},
            {detail::bilinear(-ct.c_hat, sk.f, n, batch), n},
            {-sk.s1, 1, batch.keys},
            {sk.s2, 1, batch.keys},
        });
        const auto qs = concat<G2, E>(batch.segments, {
            {ct.d, n, batch.ciphertexts},
            {ct.d_hat, n, batch.ciphertexts},
            {pk.b, n, Spread::shared},
            {ct.e, 1, batch.ciphertexts},
            {ct.e_hat, 1, batch.ciphertexts},
        });
        return table.find(pair_segments(ps, qs, PairShape{batch.segments, 3 * n + 2}));
    }
}

#endif
