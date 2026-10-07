#ifndef CUDAPFE_CURVE_CURVE_HPP
#define CUDAPFE_CURVE_CURVE_HPP

#include <array>
#include <cstddef>
#include <type_traits>
#include <cudapfe/core.hpp>
#include "field/constants.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/hd.hpp"
#include "support/words.hpp"

namespace cudapfe::detail{
    template <class F>
    struct Affine{
        F x, y;

        [[nodiscard]] CUDAPFE_HD static Affine identity(){ return {F::zero(), F::zero()}; }

        [[nodiscard]] CUDAPFE_HD bool is_identity() const{ return x.is_zero() && y.is_zero(); }

        friend CUDAPFE_HD bool operator==(const Affine& p, const Affine& q){ return p.x == q.x && p.y == q.y; }
    };

    inline constexpr std::size_t kMulWindowBits = 4;

    template <class F>
    struct Jacobian{
        F x, y, z;

        [[nodiscard]] CUDAPFE_HD static Jacobian identity(){ return {F::zero(), F::zero(), F::zero()}; }

        [[nodiscard]] CUDAPFE_HD bool is_identity() const{ return z.is_zero(); }
    };

    using G1Affine = Affine<Fp>;
    using G1Jacobian = Jacobian<Fp>;
    using G2Affine = Affine<Fp2>;
    using G2Jacobian = Jacobian<Fp2>;

    template <Side S>
    using SideField = std::conditional_t<S == Side::g1, Fp, Fp2>;

    static_assert(sizeof(G1Jacobian) == 144 && std::is_trivially_copyable_v<G1Jacobian>);
    static_assert(sizeof(G2Jacobian) == 288 && std::is_trivially_copyable_v<G2Jacobian>);

    template <class F>
    struct CurveParams;

    template <>
    struct CurveParams<Fp>{
        [[nodiscard]] CUDAPFE_HD static Fp b(){
            constexpr auto words = kCurveB;
            return Fp::from_montgomery(words);
        }

        [[nodiscard]] CUDAPFE_HD static G1Affine generator(){
            constexpr auto words = kG1Generator;
            return {Fp::from_montgomery(words[0]), Fp::from_montgomery(words[1])};
        }
    };

    template <>
    struct CurveParams<Fp2>{
        [[nodiscard]] CUDAPFE_HD static Fp2 b(){
            constexpr auto words = kCurveB;
            const auto four = Fp::from_montgomery(words);
            return {four, four};
        }

        [[nodiscard]] CUDAPFE_HD static G2Affine generator(){
            constexpr auto words = kG2Generator;
            return {Fp2::from_montgomery(words[0]), Fp2::from_montgomery(words[1])};
        }
    };

    template <class F>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> from_affine(const Affine<F>& p){
        if (p.is_identity()) return Jacobian<F>::identity();
        return {p.x, p.y, F::one()};
    }

    template <class F>
    [[nodiscard]] CUDAPFE_HD Affine<F> to_affine(const Jacobian<F>& p){
        if (p.is_identity()) return Affine<F>::identity();
        const auto z_inverse = p.z.inverse();
        const auto zz = z_inverse.square();
        return {p.x * zz, p.y * zz * z_inverse};
    }

    template <class F>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> neg(const Jacobian<F>& p){ return {p.x, -p.y, p.z}; }

    template <class F>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> dbl(const Jacobian<F>& p){
        const auto a = p.x.square();
        const auto b = p.y.square();
        const auto c = b.square();
        const auto d = (p.x + b).square() - a - c;
        const auto d2 = d + d;
        const auto e = a + a + a;
        const auto x3 = e.square() - d2 - d2;
        const auto c2 = c + c;
        const auto c4 = c2 + c2;
        const auto yz = p.y * p.z;
        return {x3, e * (d2 - x3) - (c4 + c4), yz + yz};
    }

    template <class F>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> add(const Jacobian<F>& p, const Jacobian<F>& q){
        if (p.is_identity()) return q;
        if (q.is_identity()) return p;
        const auto z1z1 = p.z.square();
        const auto z2z2 = q.z.square();
        const auto u1 = p.x * z2z2;
        const auto s1 = p.y * q.z * z2z2;
        const auto h = q.x * z1z1 - u1;
        const auto s = q.y * p.z * z1z1 - s1;
        if (h.is_zero()) return s.is_zero() ? dbl(p) : Jacobian<F>::identity();
        const auto r = s + s;
        const auto i = (h + h).square();
        const auto j = h * i;
        const auto v = u1 * i;
        const auto x3 = r.square() - j - v - v;
        const auto s1j = s1 * j;
        return {x3, r * (v - x3) - s1j - s1j, ((p.z + q.z).square() - z1z1 - z2z2) * h};
    }

    template <class F>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> add_mixed(const Jacobian<F>& p, const Affine<F>& q){
        if (q.is_identity()) return p;
        if (p.is_identity()) return from_affine(q);
        const auto z1z1 = p.z.square();
        const auto h = q.x * z1z1 - p.x;
        const auto s = q.y * p.z * z1z1 - p.y;
        if (h.is_zero()) return s.is_zero() ? dbl(p) : Jacobian<F>::identity();
        const auto hh = h.square();
        const auto hh2 = hh + hh;
        const auto i = hh2 + hh2;
        const auto j = h * i;
        const auto v = p.x * i;
        const auto r = s + s;
        const auto x3 = r.square() - j - v - v;
        const auto y1j = p.y * j;
        return {x3, r * (v - x3) - y1j - y1j, (p.z + h).square() - z1z1 - hh};
    }

    template <class F>
    [[nodiscard]] CUDAPFE_HD bool equal(const Jacobian<F>& p, const Jacobian<F>& q){
        if (p.is_identity() || q.is_identity()) return p.is_identity() && q.is_identity();
        const auto z1z1 = p.z.square();
        const auto z2z2 = q.z.square();
        return p.x * z2z2 == q.x * z1z1 && p.y * q.z * z2z2 == q.y * p.z * z1z1;
    }

    template <class F>
    [[nodiscard]] CUDAPFE_HD bool on_curve(const Affine<F>& p){
        return p.y.square() == p.x.square() * p.x + CurveParams<F>::b();
    }

    template <class F, std::size_t M>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> mul(const Jacobian<F>& p, const Words<M>& scalar){
        auto result = Jacobian<F>::identity();
#pragma unroll 1
        for (auto i = bit_length(scalar); i-- > 0;){
            result = dbl(result);
            if (bit(scalar, i)) result = add(result, p);
        }
        return result;
    }

    template <class F, std::size_t K, std::size_t M, class Map>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> interleaved_mul(
        const Jacobian<F>& p, const std::array<Words<M>, K>& digits, const Map& next
    ){
        constexpr std::size_t entries = (std::size_t{1} << kMulWindowBits) - 1;
        std::array<std::array<Jacobian<F>, entries>, K> table;
        table[0][0] = p;
#pragma unroll 1
        for (std::size_t j = 1; j < entries; ++j) table[0][j] = add(table[0][j - 1], p);
        for (std::size_t k = 1; k < K; ++k){
            for (std::size_t j = 0; j < entries; ++j) table[k][j] = next(table[k - 1][j]);
        }
        std::size_t length = 0;
        for (const auto& scalar : digits){
            const auto bits = bit_length(scalar);
            if (bits > length) length = bits;
        }
        auto result = Jacobian<F>::identity();
#pragma unroll 1
        for (auto window = ceil_div(length, kMulWindowBits); window-- > 0;){
            for (std::size_t i = 0; i < kMulWindowBits; ++i) result = dbl(result);
            const auto bit = window * kMulWindowBits;
            for (std::size_t k = 0; k < K; ++k){
                const auto digit = (digits[k][bit / 64] >> (bit % 64)) & entries;
                if (digit != 0) result = add(result, table[k][digit - 1]);
            }
        }
        return result;
    }

    [[nodiscard]] CUDAPFE_HD G1Jacobian times_z_squared(const G1Jacobian& p){
        constexpr auto words = kCubeRootOfUnitySquared;
        return {p.x * Fp::from_montgomery(words), -p.y, p.z};
    }

    [[nodiscard]] CUDAPFE_HD G1Jacobian mul(const G1Jacobian& p, const Fr& k){
        constexpr auto z_squared = kZSquared;
        const auto [hi, lo] = divide(k.canonical(), z_squared);
        const std::array<Words<2>, 2> digits{lo, Words<2>{hi[0], hi[1]}};
        return interleaved_mul(p, digits, [](const G1Jacobian& q){ return times_z_squared(q); });
    }

    [[nodiscard]] CUDAPFE_HD G2Jacobian psi(const G2Jacobian& p){
        constexpr auto x = kPsiX;
        constexpr auto y = kPsiY;
        return {conjugate(p.x) * Fp2::from_montgomery(x), conjugate(p.y) * Fp2::from_montgomery(y), conjugate(p.z)};
    }

    [[nodiscard]] CUDAPFE_HD G2Jacobian mul(const G2Jacobian& p, const Fr& k){
        constexpr Words<1> z{kZ};
        const auto [q1, d0] = divide(k.canonical(), z);
        const auto [q2, d1] = divide(q1, z);
        const auto [d3, d2] = divide(q2, z);
        const std::array<Words<1>, 4> digits{d0, d1, d2, Words<1>{d3[0]}};
        return interleaved_mul(p, digits, [](const G2Jacobian& q){ return neg(psi(q)); });
    }

    [[nodiscard]] CUDAPFE_HD bool in_subgroup(const G1Affine& p){
        const auto q = from_affine(p);
        return equal(mul(mul(q, Words<1>{kZ}), Words<1>{kZ}), times_z_squared(q));
    }

    [[nodiscard]] CUDAPFE_HD bool in_subgroup(const G2Affine& p){
        const auto q = from_affine(p);
        return equal(psi(q), neg(mul(q, Words<1>{kZ})));
    }

    template <class F>
    CUDAPFE_HD void to_affine(const Jacobian<F>* points, const std::size_t count, Affine<F>* out){
        auto running = F::one();
#pragma unroll 1
        for (std::size_t i = 0; i < count; ++i){
            out[i].x = running;
            if (!points[i].is_identity()) running = running * points[i].z;
        }
        auto inverse = running.inverse();
#pragma unroll 1
        for (std::size_t i = count; i-- > 0;){
            const auto& p = points[i];
            if (p.is_identity()){
                out[i] = Affine<F>::identity();
                continue;
            }
            const auto z_inverse = inverse * out[i].x;
            inverse = inverse * p.z;
            const auto zz = z_inverse.square();
            out[i] = {p.x * zz, p.y * zz * z_inverse};
        }
    }
}

#endif
