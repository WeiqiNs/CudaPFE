#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <blst.h>
#include <cudapfe/core.hpp>
#include <cudapfe/vector.hpp>
#include "support/access.hpp"

namespace cudapfe{
    namespace{
        using Digest = std::array<std::uint8_t, 32>;

        constexpr std::size_t kParallelFrom = 256;

        Digest sha256(const ByteView bytes){
            Digest digest;
            blst_sha256(digest.data(), bytes.data(), bytes.size());
            return digest;
        }

        Digest entropy_key(){
            std::random_device device;
            std::array<std::uint32_t, 8> words;
            for (auto& word : words) word = device();
            return std::bit_cast<Digest>(words);
        }

        struct Stream{
            Digest key;
            std::uint64_t counter;
        };

        Digest counter_block(const Digest& key, const std::uint64_t counter){
            std::array<std::uint8_t, 40> input;
            std::ranges::copy(key, input.begin());
            for (std::size_t i = 0; i < 8; ++i) input[key.size() + i] = static_cast<std::uint8_t>(counter >> (8 * i));
            return sha256(input);
        }

        Zp scalar_at(const Stream& stream, const std::uint64_t index){
            const auto low = counter_block(stream.key, stream.counter + 2 * index);
            const auto high = counter_block(stream.key, stream.counter + 2 * index + 1);
            std::array<std::uint8_t, 64> wide;
            std::ranges::copy(low, wide.begin());
            std::ranges::copy(high, wide.begin() + low.size());
            blst_scalar scalar;
            blst_scalar_from_le_bytes(&scalar, wide.data(), wide.size());
            blst_fr element;
            blst_fr_from_scalar(&element, &scalar);
            return detail::Access::value<Zp>(element);
        }

        class Generator{
        public:
            void seed(const ByteView bytes){
                const std::lock_guard lock(mutex_);
                key_ = sha256(bytes);
                counter_ = 0;
            }

            [[nodiscard]] Stream reserve(const std::uint64_t scalars){
                const std::lock_guard lock(mutex_);
                if (!key_) key_ = entropy_key();
                const Stream stream{*key_, counter_};
                counter_ += 2 * scalars;
                return stream;
            }

        private:
            std::mutex mutex_;
            std::optional<Digest> key_;
            std::uint64_t counter_ = 0;
        };

        Generator& generator(){
            static Generator instance;
            return instance;
        }
    }

    void seed(const ByteView bytes){
        if (bytes.empty()) throw ShapeError("seed needs at least one byte");
        generator().seed(bytes);
    }

    Zp Zp::random(){
        return scalar_at(generator().reserve(1), 0);
    }

    Vector random_vector(const std::size_t size){
        const auto stream = generator().reserve(size);
        Vector r(size);
#pragma omp parallel for if(size >= kParallelFrom)
        for (std::size_t i = 0; i < size; ++i) r[i] = scalar_at(stream, i);
        return r;
    }
}
