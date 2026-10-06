#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <cudapfe/core.hpp>
#include "field/constants.hpp"
#include "field/field.hpp"
#include "support/access.hpp"
#include "support/words.hpp"

namespace cudapfe{
    using detail::Access;
    using detail::Fr;

    namespace{
        Fr fr(const Zp& x){ return Access::element<Fr>(x); }

        Zp zp(const Fr& x){ return Access::value<Zp>(x); }
    }

    Zp Zp::from_unsigned(const std::uint64_t value){
        return zp(Fr::from_canonical({value}));
    }

    Zp Zp::from_signed(const std::int64_t value){
        const auto magnitude = static_cast<std::uint64_t>(value);
        return value < 0 ? -from_unsigned(0 - magnitude) : from_unsigned(magnitude);
    }

    Zp Zp::from_bytes(const ByteView bytes){
        if (bytes.size() != byte_size) throw DecodeError("Zp encoding must be exactly byte_size bytes");
        const auto words = detail::read_big_endian<4>(bytes);
        constexpr auto order = detail::kR;
        if (!detail::less(words, order)) throw DecodeError("Zp encoding is not below the group order");
        return zp(Fr::from_canonical(words));
    }

    Bytes Zp::to_bytes() const{
        Bytes out;
        out.reserve(byte_size);
        detail::append_big_endian(out, fr(*this).canonical());
        return out;
    }

    std::string Zp::to_string() const{
        const auto words = fr(*this).canonical();
        std::array<std::uint32_t, 2 * Fr::N> limbs;
        for (std::size_t i = 0; i < words.size(); ++i){
            limbs[2 * i] = static_cast<std::uint32_t>(words[i]);
            limbs[2 * i + 1] = static_cast<std::uint32_t>(words[i] >> 32);
        }
        const auto nonzero = [](const std::uint32_t limb){ return limb != 0; };
        std::string digits;
        do{
            std::uint64_t remainder = 0;
            for (auto limb = limbs.rbegin(); limb != limbs.rend(); ++limb){
                const auto current = remainder << 32 | *limb;
                *limb = static_cast<std::uint32_t>(current / 10);
                remainder = current % 10;
            }
            digits.push_back(static_cast<char>('0' + remainder));
        } while (std::any_of(limbs.begin(), limbs.end(), nonzero));
        std::reverse(digits.begin(), digits.end());
        return digits;
    }

    bool Zp::is_zero() const{
        return fr(*this).is_zero();
    }

    Zp Zp::inverse() const{
        if (is_zero()) throw NotInvertible("zero has no inverse in Zp");
        return zp(fr(*this).inverse());
    }

    Zp Zp::pow(const std::uint64_t exponent) const{
        return zp(fr(*this).pow(detail::Words<1>{exponent}));
    }

    Zp Zp::plus(const Zp& y) const{
        return zp(fr(*this) + fr(y));
    }

    Zp Zp::minus(const Zp& y) const{
        return zp(fr(*this) - fr(y));
    }

    Zp Zp::times(const Zp& y) const{
        return zp(fr(*this) * fr(y));
    }

    Zp Zp::negated() const{
        return zp(-fr(*this));
    }
}
