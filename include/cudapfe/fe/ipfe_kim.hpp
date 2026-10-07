#ifndef CUDAPFE_FE_IPFE_KIM_HPP
#define CUDAPFE_FE_IPFE_KIM_HPP

#include <cstddef>
#include <utility>
#include <vector>
#include "ipfe.hpp"

namespace cudapfe::IPFE::KIM{
    using IPFE::dec, IPFE::enc, IPFE::keygen;

    template <Engine E>
    struct Msk{
        Zp det;
        Matrix<E> b;
        Matrix<E> bi;
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
        Vec<G2, E> r;
        G2Lines<E> vec;
    };

    template <Engine E>
    [[nodiscard]] Msk<E> setup(const std::size_t size){
        detail::require_setup_memory<E>(size, "IPFE::KIM");
        auto b = Matrix<E>::random({size, size});
        auto [inverse, det] = b.inverse_with_determinant();
        return {det, std::move(b), (inverse * det).transpose()};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntMatrix& functions){
        Vector r;
        std::vector<Vector> encoded;
        for (const auto& function : functions){
            const auto alpha = Zp::random();
            r.push_back(alpha * msk.det);
            encoded.push_back(to_vector(function) * alpha);
        }
        return {functions.size(), detail::lift<G2, E>(r), detail::lift<G2>(encoded, msk.b)};
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntMatrix& messages){
        Vector r;
        std::vector<Vector> encoded;
        for (const auto& message : messages){
            const auto beta = Zp::random();
            r.push_back(beta);
            encoded.push_back(to_vector(message) * beta);
        }
        return {messages.size(), detail::lift<G1, E>(r), detail::lift<G1>(encoded, msk.bi)};
    }

    template <Engine E>
    [[nodiscard]] PreparedSk<E> prepare(const Sk<E>& sk){
        return {sk.count, sk.r, cudapfe::prepare(sk.vec)};
    }
}

#endif
