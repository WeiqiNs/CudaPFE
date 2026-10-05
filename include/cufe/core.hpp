#ifndef CUFE_CORE_HPP
#define CUFE_CORE_HPP

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>
#include "errors.hpp"

namespace cufe{
    using Bytes = std::vector<std::uint8_t>;
    using ByteView = std::span<const std::uint8_t>;

    enum class Encoding{ compressed, uncompressed };

    [[nodiscard]] inline Bytes bytes_of(const std::string_view text){
        return {text.begin(), text.end()};
    }
}

#endif
