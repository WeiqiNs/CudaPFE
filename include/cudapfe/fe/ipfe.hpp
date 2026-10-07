#ifndef CUDAPFE_FE_IPFE_HPP
#define CUDAPFE_FE_IPFE_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string_view>
#include <vector>
#include <cudapfe/cudapfe.hpp>

namespace cudapfe::IPFE{
    using IntVec = std::vector<std::int64_t>;
    using IntMatrix = std::vector<IntVec>;
    using Results = std::vector<std::optional<std::int64_t>>;

    [[nodiscard]] inline Vector to_vector(const IntVec& x){
        return {x.begin(), x.end()};
    }

    [[nodiscard]] inline Gt base(){
        return Gt::generator();
    }

    namespace detail{
        struct Counts{
            std::size_t keys;
            std::size_t ciphertexts;
        };

        struct Broadcast{
            std::size_t segments;
            Spread keys;
            Spread ciphertexts;
        };

        [[nodiscard]] inline Broadcast broadcast(const Counts& counts){
            if (counts.keys == counts.ciphertexts) return {counts.keys, Spread::per_segment, Spread::per_segment};
            if (counts.keys == 1) return {counts.ciphertexts, Spread::shared, Spread::per_segment};
            if (counts.ciphertexts == 1) return {counts.keys, Spread::per_segment, Spread::shared};
            throw ShapeError(std::format(
                "dec needs as many keys as ciphertexts, or one of either, got {} keys and {} ciphertexts", counts.keys,
                counts.ciphertexts
            ));
        }

        inline void append(Vector& out, const Vector& values){
            out.insert(out.end(), values.begin(), values.end());
        }

        template <Engine E, class Keys>
        [[nodiscard]] Vec<Gt, E> pair_entries(const Vec<G1, E>& cts, const Keys& keys, const Counts& counts){
            const auto batch = broadcast(counts);
            if (batch.segments == 0) return {};
            const auto length = cts.size() / (batch.ciphertexts == Spread::shared ? 1 : batch.segments);
            return pair_segments(cts, keys, PairShape{batch.segments, length, batch.ciphertexts, batch.keys});
        }

        template <class G, Engine E>
        [[nodiscard]] Vec<G, E> lift(const Vector& exponents){
            return mul_generator<G>(Vec<Zp, E>::upload(exponents));
        }

        template <class G, Engine E>
        [[nodiscard]] Vec<G, E> lift(const std::vector<Vector>& rows, const Matrix<E>& basis){
            if (rows.empty()) return {};
            return mul_generator<G>((Matrix<E>::from_rows(rows) * basis).entries());
        }

        template <Engine E>
        void require_setup_memory(const std::size_t size, const std::string_view scheme){
            if constexpr (std::same_as<E, Gpu>){
                constexpr double kLiveMatrices = 4;
                const auto matrix = static_cast<double>(size) * static_cast<double>(size) * Zp::byte_size;
                const auto needed = kLiveMatrices * matrix;
                const auto free = static_cast<double>(gpu_free_memory());
                if (needed > free){
                    throw DeviceError(std::format(
                        "{} setup inverts a {} x {} matrix, which needs about {:.1f} GB of device memory, but "
                        "{:.1f} GB is free", scheme, size, size, needed / 1e9, free / 1e9
                    ));
                }
            }
        }
    }

    template <class Msk>
    [[nodiscard]] auto keygen(const Msk& msk, const IntVec& function){
        return keygen(msk, IntMatrix{function});
    }

    template <class Msk>
    [[nodiscard]] auto enc(const Msk& msk, const IntVec& message){
        return enc(msk, IntMatrix{message});
    }

    template <Engine E, class Key, class Cipher>
    [[nodiscard]] Results dec(const DlogTable<E>& table, const Key& sk, const Cipher& ct){
        return table.find(detail::pair_entries(ct.vec, sk.vec, {.keys = sk.count, .ciphertexts = ct.count}));
    }

    template <class Key, class Cipher>
    [[nodiscard]] Results dec(const Key& sk, const Cipher& ct, const Range& range){
        const detail::Counts counts{.keys = sk.count, .ciphertexts = ct.count};
        const DlogTables tables(detail::pair_entries(ct.r, sk.r, counts), range);
        return tables.find(detail::pair_entries(ct.vec, sk.vec, counts));
    }
}

#endif
