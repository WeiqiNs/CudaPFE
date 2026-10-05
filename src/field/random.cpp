#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <blst.h>
#include <cufe/core.hpp>
#include "support/access.hpp"

namespace cufe{
    namespace{
        using Digest = std::array<std::uint8_t, 32>;

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

        class Generator{
        public:
            void seed(const ByteView bytes){
                const std::lock_guard lock(mutex_);
                key_ = sha256(bytes);
                counter_ = 0;
            }

            [[nodiscard]] std::array<std::uint8_t, 64> wide_block(){
                const std::lock_guard lock(mutex_);
                if (!key_) key_ = entropy_key();
                const auto low = next_block();
                const auto high = next_block();
                std::array<std::uint8_t, 64> block;
                std::copy(low.begin(), low.end(), block.begin());
                std::copy(high.begin(), high.end(), block.begin() + low.size());
                return block;
            }

        private:
            Digest next_block(){
                std::array<std::uint8_t, 40> input;
                std::copy(key_->begin(), key_->end(), input.begin());
                for (std::size_t i = 0; i < 8; ++i){
                    input[key_->size() + i] = static_cast<std::uint8_t>(counter_ >> (8 * i));
                }
                ++counter_;
                return sha256(input);
            }

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
        const auto block = generator().wide_block();
        blst_scalar scalar;
        blst_scalar_from_le_bytes(&scalar, block.data(), block.size());
        blst_fr element;
        blst_fr_from_scalar(&element, &scalar);
        return detail::Access::value<Zp>(element);
    }
}
