#include <array>
#include <bit>
#include <cstddef>
#include <cufe/core.hpp>
#include "curve/encoding.hpp"
#include "field/constants.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/access.hpp"

namespace cufe{
    using detail::Access;
    using detail::Fp;
    using detail::Fp2;
    using detail::Fp6;
    using detail::Fp12;

    namespace{
        struct Coefficient{
            Fp6 Fp12::* half;
            Fp2 Fp6::* slot;
        };

        constexpr std::array<Coefficient, 6> kBlstOrder{{
            {&Fp12::c0, &Fp6::c0}, {&Fp12::c1, &Fp6::c0}, {&Fp12::c0, &Fp6::c1},
            {&Fp12::c1, &Fp6::c1}, {&Fp12::c0, &Fp6::c2}, {&Fp12::c1, &Fp6::c2},
        }};

        Fp12 fp12(const Gt& x){ return Access::element<Fp12>(x); }

        Gt gt(const Fp12& x){ return Access::value<Gt>(x); }
    }

    Gt::Gt() : words_(std::bit_cast<decltype(words_)>(Fp12::one())){}

    Gt Gt::generator(){
        static const Gt generator = pair(G1::generator(), G2::generator());
        return generator;
    }

    Gt Gt::random(){
        return generator().pow(Zp::random());
    }

    Gt Gt::from_bytes(const ByteView bytes){
        if (bytes.size() != byte_size) throw DecodeError("Gt encoding must be exactly byte_size bytes");
        constexpr auto size = detail::kFpBytes;
        Fp12 f;
        auto rest = bytes;
        for (const auto& [half, slot] : kBlstOrder){
            f.*half.*slot = {detail::read_fp(rest.first(size)), detail::read_fp(rest.subspan(size, size))};
            rest = rest.subspan(2 * size);
        }
        constexpr auto order = detail::kR;
        if (!(detail::pow(f, order) == Fp12::one())) throw DecodeError("Gt encoding is not in the order-r subgroup");
        return gt(f);
    }

    Bytes Gt::to_bytes() const{
        const auto f = fp12(*this);
        Bytes out;
        out.reserve(byte_size);
        for (const auto& [half, slot] : kBlstOrder){
            const auto& coefficient = f.*half.*slot;
            detail::append(out, coefficient.c0);
            detail::append(out, coefficient.c1);
        }
        return out;
    }

    bool Gt::is_one() const{
        return fp12(*this) == Fp12::one();
    }

    Gt Gt::inverse() const{
        return gt(detail::conjugate(fp12(*this)));
    }

    Gt Gt::pow(const Zp& exponent) const{
        return gt(detail::pow(fp12(*this), Access::element<detail::Fr>(exponent).canonical()));
    }

    Gt Gt::times(const Gt& y) const{
        return gt(fp12(*this) * fp12(y));
    }
}
