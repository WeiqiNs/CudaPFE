#ifndef CUFE_FE_QFE_HPP
#define CUFE_FE_QFE_HPP

#include <cstddef>
#include <format>
#include <string_view>
#include <vector>
#include "ipfe.hpp"

namespace cufe::QFE{
    using IPFE::IntMatrix;
    using IPFE::IntVec;
    using IPFE::to_vector;

    namespace detail{
        using IPFE::detail::append;
        using IPFE::detail::broadcast;
        using IPFE::detail::Broadcast;
        using IPFE::detail::lift;

        struct Mask{
            Vector r;
            Vector m;

            void add(const Zp& randomness, const Vector& message){
                r.push_back(randomness);
                append(m, message);
            }
        };

        [[nodiscard]] inline Vector row(const IntVec& values, const std::size_t size, const std::string_view operation){
            if (values.size() != size){
                throw ShapeError(std::format("{} needs vectors of length {}, got {}", operation, size, values.size()));
            }
            return to_vector(values);
        }

        [[nodiscard]] inline std::vector<Vector> rows(
            const IntMatrix& f, const std::size_t size, const std::string_view operation
        ){
            if (f.size() != size){
                throw ShapeError(
                    std::format("{} needs {} x {} matrices, got {} rows", operation, size, size, f.size())
                );
            }
            std::vector<Vector> result;
            for (const auto& values : f) result.push_back(row(values, size, operation));
            return result;
        }

        [[nodiscard]] inline Zp quadratic_form(const Vector& x, const std::vector<Vector>& f, const Vector& y){
            Zp total;
            for (std::size_t i = 0; i < f.size(); ++i) total += x[i] * inner(f[i], y);
            return total;
        }

        inline void require_pairs(const IntMatrix& lefts, const IntMatrix& rights, const std::string_view operation){
            if (lefts.size() != rights.size()){
                throw ShapeError(std::format(
                    "{} needs one right vector per left vector, got {} and {}", operation, lefts.size(), rights.size()
                ));
            }
        }

        [[nodiscard]] inline Vector repeat(const Vector& values, const std::size_t times){
            Vector repeated;
            for (const auto& value : values) repeated.insert(repeated.end(), times, value);
            return repeated;
        }

        template <class G, Engine E>
        [[nodiscard]] Vec<G, E> scaled_copies(const Vec<G, E>& base, const Vector& factors){
            return concat<G, E>(factors.size(), {{base, base.size(), Spread::shared}})
                * Vec<Zp, E>::upload(repeat(factors, base.size()));
        }

        template <class G, Engine E>
        [[nodiscard]] Vec<G, E> masked(const Vec<G, E>& base, const Mask& mask){
            return lift<G, E>(mask.m) + scaled_copies(base, mask.r);
        }

        template <Engine E>
        [[nodiscard]] Vec<G1, E> bilinear(
            const Vec<G1, E>& ciphertext_points, const Vec<Zp, E>& f, const std::size_t size, const Broadcast& batch
        ){
            return msm(ciphertext_points, f, MsmShape{batch.segments, {size, size}, batch.ciphertexts, batch.keys});
        }
    }
}

#endif
