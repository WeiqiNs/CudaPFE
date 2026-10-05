#ifndef CUFE_FE_QFE_SGP_HPP
#define CUFE_FE_QFE_SGP_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "qfe.hpp"

namespace cufe::QFE::SGP{
    template <Engine E>
    struct Msk{
        Vector s;
        Vector t;
    };

    template <Engine E>
    struct Pk{
        Vec<G1, E> s;
        Vec<G2, E> t;
    };

    template <Engine E>
    struct Keys{
        Pk<E> pk;
        Msk<E> msk;
    };

    template <Engine E>
    struct Sk{
        Vec<Zp, E> f;
        Vec<G2, E> key;
    };

    template <Engine E>
    struct Ct{
        Vec<G1, E> gamma;
        Vec<G1, E> a0;
        Vec<G1, E> a1;
        Vec<G2, E> b0;
        Vec<G2, E> b1;
    };

    template <Engine E>
    [[nodiscard]] Keys<E> setup(const std::size_t size){
        auto s = random_vector(size);
        auto t = random_vector(size);
        return {{detail::lift<G1, E>(s), detail::lift<G2, E>(t)}, {std::move(s), std::move(t)}};
    }

    [[nodiscard]] inline Gt base(){
        return Gt::generator();
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const std::vector<IntMatrix>& functions){
        Vector f;
        Vector keys;
        for (const auto& function : functions){
            const auto rows = detail::rows(function, msk.s.size(), "QFE::SGP::keygen");
            for (const auto& row : rows) detail::append(f, row);
            keys.push_back(detail::quadratic_form(msk.s, rows, msk.t));
        }
        return {Vec<Zp, E>::upload(f), detail::lift<G2, E>(keys)};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntMatrix& function){
        return keygen(msk, std::vector<IntMatrix>{function});
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Pk<E>& pk, const IntMatrix& lefts, const IntMatrix& rights){
        detail::require_pairs(lefts, rights, "QFE::SGP::enc");
        Vector gammas;
        detail::Mask a0;
        detail::Mask a1;
        detail::Mask b0;
        detail::Mask b1;
        for (std::size_t i = 0; i < lefts.size(); ++i){
            const auto x = detail::row(lefts[i], pk.s.size(), "QFE::SGP::enc");
            const auto y = detail::row(rights[i], pk.t.size(), "QFE::SGP::enc");
            const auto gamma = Zp::random();
            const auto w = Matrix<Cpu>::random({2, 2});
            const auto wi = w.inverse().transpose().to_rows();
            const auto wr = w.to_rows();
            gammas.push_back(gamma);
            a0.add(gamma * wi[0][1], x * wi[0][0]);
            a1.add(gamma * wi[1][1], x * wi[1][0]);
            b0.add(-wr[0][1], y * wr[0][0]);
            b1.add(-wr[1][1], y * wr[1][0]);
        }
        return {
            detail::lift<G1, E>(gammas),
            detail::masked(pk.s, a0),
            detail::masked(pk.s, a1),
            detail::masked(pk.t, b0),
            detail::masked(pk.t, b1)
        };
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Pk<E>& pk, const IntVec& left, const IntVec& right){
        return enc(pk, IntMatrix{left}, IntMatrix{right});
    }

    template <Engine E>
    [[nodiscard]] std::vector<std::optional<std::int64_t>> dec(
        const DlogTable<E>& table, const Sk<E>& sk, const Ct<E>& ct
    ){
        const auto batch = detail::broadcast({.keys = sk.key.size(), .ciphertexts = ct.gamma.size()});
        if (batch.segments == 0) return {};
        const auto n = ct.a0.size() / ct.gamma.size();
        const auto ps = concat<G1, E>(batch.segments, {
            {ct.gamma, 1, batch.ciphertexts},
            {detail::bilinear(ct.a0, sk.f, n, batch), n},
            {detail::bilinear(ct.a1, sk.f, n, batch), n},
        });
        const auto qs = concat<G2, E>(batch.segments, {
            {sk.key, 1, batch.keys},
            {ct.b0, n, batch.ciphertexts},
            {ct.b1, n, batch.ciphertexts},
        });
        return table.find(pair_segments(ps, qs, PairShape{batch.segments, 2 * n + 1}));
    }
}

#endif
