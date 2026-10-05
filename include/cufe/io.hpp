#ifndef CUFE_IO_HPP
#define CUFE_IO_HPP

#include <format>
#include <ostream>
#include <string>
#include <string_view>
#include "core.hpp"

namespace cufe{
    [[nodiscard]] inline std::string to_hex(const ByteView bytes){
        constexpr std::string_view digits = "0123456789abcdef";
        std::string out;
        out.reserve(2 * bytes.size());
        for (const auto byte : bytes){
            out.push_back(digits[byte >> 4]);
            out.push_back(digits[byte & 0x0f]);
        }
        return out;
    }

    inline std::ostream& operator<<(std::ostream& out, const Zp& x){
        return out << x.to_string();
    }

    template <Side S>
    std::ostream& operator<<(std::ostream& out, const Point<S>& p){
        return out << to_hex(p.to_bytes());
    }

    inline std::ostream& operator<<(std::ostream& out, const Gt& x){
        return out << to_hex(x.to_bytes());
    }
}

template <>
struct std::formatter<cufe::Zp> : std::formatter<std::string>{
    auto format(const cufe::Zp& x, std::format_context& context) const{
        return std::formatter<std::string>::format(x.to_string(), context);
    }
};

template <cufe::Side S>
struct std::formatter<cufe::Point<S>> : std::formatter<std::string>{
    auto format(const cufe::Point<S>& p, std::format_context& context) const{
        return std::formatter<std::string>::format(cufe::to_hex(p.to_bytes()), context);
    }
};

template <>
struct std::formatter<cufe::Gt> : std::formatter<std::string>{
    auto format(const cufe::Gt& x, std::format_context& context) const{
        return std::formatter<std::string>::format(cufe::to_hex(x.to_bytes()), context);
    }
};

#endif
