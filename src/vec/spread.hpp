#ifndef CUDAPFE_VEC_SPREAD_HPP
#define CUDAPFE_VEC_SPREAD_HPP

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <cudapfe/errors.hpp>
#include <cudapfe/vec.hpp>
#include "support/hd.hpp"

namespace cudapfe::detail{
    [[nodiscard]] inline std::size_t checked_product(const std::size_t x, const std::size_t y, const std::string_view what){
        if (x != 0 && y > std::numeric_limits<std::size_t>::max() / x){
            throw ShapeError(std::string(what) + " shape of " + std::to_string(x) + " x " + std::to_string(y)
                + " overflows");
        }
        return x * y;
    }

    struct Layout{
        std::size_t segments;
        std::size_t length;
        Spread spread;

        [[nodiscard]] CUDAPFE_HD constexpr std::size_t offset(const std::size_t segment) const{
            return spread == Spread::shared ? 0 : segment * length;
        }

        void require_size(const std::size_t size, const std::string_view what) const{
            const auto expected = spread == Spread::shared ? length : checked_product(segments, length, what);
            if (size != expected){
                throw ShapeError(std::string(what) + " needs " + std::to_string(expected) + " for its shape, got "
                    + std::to_string(size));
            }
        }
    };
}

#endif
