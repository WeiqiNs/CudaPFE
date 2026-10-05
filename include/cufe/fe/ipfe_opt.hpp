#ifndef CUFE_FE_IPFE_OPT_HPP
#define CUFE_FE_IPFE_OPT_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include "ipfe.hpp"

namespace cufe::IPFE::OPT{
    inline constexpr std::size_t b_size = 4;

    template <Engine E>
    struct Msk{
        std::vector<Vector> a;
        std::vector<Vector> b;
        std::vector<Vector> bi;
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
        G2Lines<E> key;
    };

    namespace detail{
        [[nodiscard]] inline Vector times(const std::vector<Vector>& rows, const Vector& x){
            Vector product;
            for (const auto& row : rows) product.push_back(inner(row, x));
            return product;
        }

        [[nodiscard]] inline Vector times(const Vector& x, const std::vector<Vector>& rows){
            if (x.size() != rows.size()) throw ShapeError("a vector-matrix product needs one entry per matrix row");
            Vector product(rows.front().size());
            for (std::size_t i = 0; i < rows.size(); ++i) product = product + rows[i] * x[i];
            return product;
        }

        [[nodiscard]] inline std::size_t entry_length(const std::size_t points, const std::size_t count){
            return count == 0 ? 0 : points / count;
        }

        template <Engine E>
        [[nodiscard]] Vec<G2, E> key_points(const Sk<E>& sk){
            const auto length = entry_length(sk.vec.size(), sk.count);
            return concat<G2, E>(sk.count, {{sk.vec, length}, {sk.r, b_size}});
        }

        template <Engine E>
        [[nodiscard]] const G2Lines<E>& key_points(const PreparedSk<E>& sk){
            return sk.key;
        }

        template <Engine E>
        [[nodiscard]] Vec<G1, E> ciphertext_points(const Ct<E>& ct){
            const auto length = entry_length(ct.vec.size(), ct.count);
            return concat<G1, E>(ct.count, {{ct.vec, length}, {-ct.r, b_size}});
        }
    }

    template <Engine E>
    [[nodiscard]] Msk<E> setup(const std::size_t size){
        const auto b = Matrix<Cpu>::random({b_size, b_size});
        return {{random_vector(size), random_vector(size)}, b.to_rows(), b.inverse().transpose().to_rows()};
    }

    [[nodiscard]] inline Gt base(){
        return Gt::generator();
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntMatrix& functions){
        Vector r;
        Vector vec;
        for (const auto& function : functions){
            const auto s = random_vector(2);
            const auto masked = detail::times(s, msk.a) + to_vector(function);
            IPFE::detail::append(r, detail::times(msk.b, concat({s, detail::times(msk.a, masked)})));
            IPFE::detail::append(vec, masked);
        }
        return {functions.size(), IPFE::detail::lift<G2, E>(r), IPFE::detail::lift<G2, E>(vec)};
    }

    template <Engine E>
    [[nodiscard]] Sk<E> keygen(const Msk<E>& msk, const IntVec& function){
        return keygen(msk, IntMatrix{function});
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntMatrix& messages){
        Vector r;
        Vector vec;
        for (const auto& message : messages){
            const auto m = to_vector(message);
            const auto s = random_vector(2);
            const auto masked = detail::times(s, msk.a) + m;
            IPFE::detail::append(r, detail::times(msk.bi, concat({detail::times(msk.a, m), s})));
            IPFE::detail::append(vec, masked);
        }
        return {messages.size(), IPFE::detail::lift<G1, E>(r), IPFE::detail::lift<G1, E>(vec)};
    }

    template <Engine E>
    [[nodiscard]] Ct<E> enc(const Msk<E>& msk, const IntVec& message){
        return enc(msk, IntMatrix{message});
    }

    template <Engine E>
    [[nodiscard]] PreparedSk<E> prepare(const Sk<E>& sk){
        return {sk.count, cufe::prepare(detail::key_points(sk))};
    }

    template <Engine E, class Key> requires std::same_as<Key, Sk<E>> || std::same_as<Key, PreparedSk<E>>
    [[nodiscard]] std::vector<std::optional<std::int64_t>> dec(
        const DlogTable<E>& table, const Key& sk, const Ct<E>& ct
    ){
        const auto batch = IPFE::detail::broadcast({.keys = sk.count, .ciphertexts = ct.count});
        return table.find(IPFE::detail::pair_entries(detail::ciphertext_points(ct), detail::key_points(sk), batch));
    }
}

#endif
