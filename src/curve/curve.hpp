#ifndef CUFE_CURVE_CURVE_HPP
#define CUFE_CURVE_CURVE_HPP

#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <vector>
#include <cufe/errors.hpp>
#include "field/constants.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/hd.hpp"
#include "support/words.hpp"

namespace cufe::detail{
    template <class F>
    struct Affine{
        F x, y;

        [[nodiscard]] CUFE_HD static Affine identity(){ return {F::zero(), F::zero()}; }

        [[nodiscard]] CUFE_HD bool is_identity() const{ return x.is_zero() && y.is_zero(); }

        friend CUFE_HD bool operator==(const Affine& p, const Affine& q){ return p.x == q.x && p.y == q.y; }
    };

    template <class F>
    struct Jacobian{
        F x, y, z;

        [[nodiscard]] CUFE_HD static Jacobian identity(){ return {F::zero(), F::zero(), F::zero()}; }

        [[nodiscard]] CUFE_HD bool is_identity() const{ return z.is_zero(); }
    };

    using G1Affine = Affine<Fp>;
    using G1Jacobian = Jacobian<Fp>;
    using G2Affine = Affine<Fp2>;
    using G2Jacobian = Jacobian<Fp2>;

    static_assert(sizeof(G1Jacobian) == 144 && std::is_trivially_copyable_v<G1Jacobian>);
    static_assert(sizeof(G2Jacobian) == 288 && std::is_trivially_copyable_v<G2Jacobian>);

    template <class F>
    struct CurveParams;

    template <>
    struct CurveParams<Fp>{
        [[nodiscard]] CUFE_HD static Fp b(){
            constexpr auto words = kCurveB;
            return Fp::from_montgomery(words);
        }

        [[nodiscard]] CUFE_HD static G1Affine generator(){
            constexpr auto words = kG1Generator;
            return {Fp::from_montgomery(words[0]), Fp::from_montgomery(words[1])};
        }
    };

    template <>
    struct CurveParams<Fp2>{
        [[nodiscard]] CUFE_HD static Fp2 b(){
            constexpr auto words = kCurveB;
            const auto four = Fp::from_montgomery(words);
            return {four, four};
        }

        [[nodiscard]] CUFE_HD static G2Affine generator(){
            constexpr auto words = kG2Generator;
            return {Fp2::from_montgomery(words[0]), Fp2::from_montgomery(words[1])};
        }
    };

    template <class F>
    [[nodiscard]] CUFE_HD Jacobian<F> from_affine(const Affine<F>& p){
        if (p.is_identity()) return Jacobian<F>::identity();
        return {p.x, p.y, F::one()};
    }

    template <class F>
    [[nodiscard]] CUFE_HD Affine<F> to_affine(const Jacobian<F>& p){
        if (p.is_identity()) return Affine<F>::identity();
        const auto z_inverse = p.z.inverse();
        const auto zz = z_inverse.square();
        return {p.x * zz, p.y * zz * z_inverse};
    }

    template <class F>
    [[nodiscard]] CUFE_HD Jacobian<F> neg(const Jacobian<F>& p){ return {p.x, -p.y, p.z}; }

    template <class F>
    [[nodiscard]] CUFE_HD Jacobian<F> dbl(const Jacobian<F>& p){
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
    [[nodiscard]] CUFE_HD Jacobian<F> add(const Jacobian<F>& p, const Jacobian<F>& q){
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
    [[nodiscard]] CUFE_HD Jacobian<F> add_mixed(const Jacobian<F>& p, const Affine<F>& q){
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
    [[nodiscard]] CUFE_HD bool equal(const Jacobian<F>& p, const Jacobian<F>& q){
        if (p.is_identity() || q.is_identity()) return p.is_identity() && q.is_identity();
        const auto z1z1 = p.z.square();
        const auto z2z2 = q.z.square();
        return p.x * z2z2 == q.x * z1z1 && p.y * q.z * z2z2 == q.y * p.z * z1z1;
    }

    template <class F>
    [[nodiscard]] CUFE_HD bool on_curve(const Affine<F>& p){
        return p.y.square() == p.x.square() * p.x + CurveParams<F>::b();
    }

    template <class F, std::size_t M>
    [[nodiscard]] CUFE_HD Jacobian<F> mul(const Jacobian<F>& p, const Words<M>& scalar){
        auto result = Jacobian<F>::identity();
#pragma unroll 1
        for (auto i = bit_length(scalar); i-- > 0;){
            result = dbl(result);
            if (bit(scalar, i)) result = add(result, p);
        }
        return result;
    }

    template <class F>
    [[nodiscard]] CUFE_HD Jacobian<F> mul(const Jacobian<F>& p, const Fr& k){ return mul(p, k.canonical()); }

    template <class F>
    [[nodiscard]] CUFE_HD bool in_subgroup(const Affine<F>& p){
        constexpr auto order = kR;
        return mul(from_affine(p), order).is_identity();
    }

    template <class F>
    void to_affine(const std::span<const Jacobian<F>> points, const std::span<Affine<F>> out){
        if (points.size() != out.size()){
            throw ShapeError("to_affine of " + std::to_string(points.size()) + " points cannot fill "
                + std::to_string(out.size()));
        }
        std::vector<F> prefix(points.size());
        auto running = F::one();
        for (std::size_t i = 0; i < points.size(); ++i){
            prefix[i] = running;
            if (!points[i].is_identity()) running = running * points[i].z;
        }
        auto inverse = running.inverse();
        for (std::size_t i = points.size(); i-- > 0;){
            const auto& p = points[i];
            if (p.is_identity()){
                out[i] = Affine<F>::identity();
                continue;
            }
            const auto z_inverse = inverse * prefix[i];
            inverse = inverse * p.z;
            const auto zz = z_inverse.square();
            out[i] = {p.x * zz, p.y * zz * z_inverse};
        }
    }
}

#endif
