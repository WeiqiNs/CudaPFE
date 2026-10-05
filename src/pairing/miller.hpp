#ifndef CUFE_PAIRING_MILLER_HPP
#define CUFE_PAIRING_MILLER_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include "curve/curve.hpp"
#include "field/constants.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/hd.hpp"

namespace cufe::detail{
    enum class Step : std::uint8_t{ dbl, add };

    using Line = Fp6;

    inline constexpr std::size_t kLineCount = std::bit_width(kZ) - 1 + std::popcount(kZ) - 1;

    consteval std::array<Step, kLineCount> make_schedule(const Word z){
        std::array<Step, kLineCount> schedule{};
        std::size_t next = 0;
        for (auto bit = std::bit_width(z) - 1; bit-- > 0;){
            schedule[next++] = Step::dbl;
            if ((z >> bit) & 1) schedule[next++] = Step::add;
        }
        return schedule;
    }

    inline constexpr std::array<Step, kLineCount> kMillerSchedule = make_schedule(kZ);


    [[nodiscard]] CUFE_HD Line line_dbl(G2Jacobian& t){
        const auto a = t.x.square();
        const auto b = t.y.square();
        const auto zz = t.z.square();
        const auto c = b.square();
        const auto d = (t.x + b).square() - a - c;
        const auto d2 = d + d;
        const auto e = a + a + a;
        const auto f = e.square();
        const auto ex = e + t.x;
        const auto x3 = f - d2 - d2;
        const auto z3 = (t.y + t.z).square() - b - zz;
        const auto c2 = c + c;
        const auto c4 = c2 + c2;
        const auto b2 = b + b;
        t = {x3, e * (d2 - x3) - (c4 + c4), z3};
        return {ex.square() - a - f - (b2 + b2), e * zz, z3 * zz};
    }

    [[nodiscard]] CUFE_HD Line line_add(G2Jacobian& t, const G2Affine& q){
        const auto z1z1 = t.z.square();
        const auto h = q.x * z1z1 - t.x;
        const auto s = q.y * t.z * z1z1 - t.y;
        const auto hh = h.square();
        const auto hh2 = hh + hh;
        const auto i = hh2 + hh2;
        const auto j = h * i;
        const auto r = s + s;
        const auto v = t.x * i;
        const auto x3 = r.square() - j - v - v;
        const auto yj = t.y * j;
        const auto z3 = (t.z + h).square() - z1z1 - hh;
        t = {x3, r * (v - x3) - yj - yj, z3};
        const auto line = r * q.x - q.y * z3;
        return {line + line, r, z3};
    }

    CUFE_HD void prepare_lines(const G2Affine& q, Line* out){
        if (q.is_identity()) return;
        constexpr auto schedule = kMillerSchedule;
        G2Jacobian t{q.x, q.y, Fp2::one()};
#pragma unroll 1
        for (std::size_t step = 0; step < kLineCount; ++step){
            out[step] = schedule[step] == Step::dbl ? line_dbl(t) : line_add(t, q);
        }
    }

    [[nodiscard]] CUFE_HD G1Affine px2(const G1Affine& p){ return {-(p.x + p.x), p.y + p.y}; }

    struct PreparedPairs{
        const G1Affine* ps;
        const G2Affine* qs;
        const Line* lines;
        std::size_t size;

        [[nodiscard]] CUFE_HD std::size_t count() const{ return size; }

        [[nodiscard]] CUFE_HD bool live(const std::size_t i) const{
            return !ps[i].is_identity() && !qs[i].is_identity();
        }

        [[nodiscard]] CUFE_HD G1Affine px2(const std::size_t i) const{ return detail::px2(ps[i]); }

        [[nodiscard]] CUFE_HD const Line& line(const std::size_t i, const std::size_t step) const{
            return lines[i * kLineCount + step];
        }
    };

    [[nodiscard]] CUFE_HD Line evaluate(const Line& line, const G1Affine& px2){
        return {line.c0, line.c1 * px2.x, line.c2 * px2.y};
    }

    template <class Pairs>
    [[nodiscard]] CUFE_HD Fp12 miller(const Pairs& pairs){
        constexpr auto schedule = kMillerSchedule;
        auto f = Fp12::one();
#pragma unroll 1
        for (std::size_t step = 0; step < kLineCount; ++step){
            if (step != 0 && schedule[step] == Step::dbl) f = f.square();
#pragma unroll 1
            for (std::size_t i = 0; i < pairs.count(); ++i){
                if (pairs.live(i)) f = mul_by_line(f, evaluate(pairs.line(i, step), pairs.px2(i)));
            }
        }
        if constexpr (kZIsNegative) return conjugate(f);
        else return f;
    }

    [[nodiscard]] CUFE_HD Fp12 raise_to_z_div_by_2(const Fp12& a){
        constexpr Word exponent = kZ >> 1;
        constexpr int top = std::bit_width(exponent) - 1;
        auto result = a;
#pragma unroll 1
        for (int bit = top; bit-- > 0;){
            result = cyclotomic_square(result);
            if ((exponent >> bit) & 1) result = result * a;
        }
        if constexpr (kZIsNegative) return conjugate(result);
        else return result;
    }

    [[nodiscard]] CUFE_HD Fp12 raise_to_z(const Fp12& a){ return cyclotomic_square(raise_to_z_div_by_2(a)); }

    [[nodiscard]] CUFE_HD Fp12 final_exp(const Fp12& f){
        auto ret = conjugate(f) * f.inverse();
        ret = ret * frobenius<2>(ret);
        const auto y0 = cyclotomic_square(ret);
        auto y1 = raise_to_z(y0);
        auto y2 = raise_to_z_div_by_2(y1);
        y1 = conjugate(y1 * conjugate(ret)) * y2;
        y2 = raise_to_z(y1);
        const auto y3 = raise_to_z(y2) * conjugate(y1);
        y1 = frobenius<3>(y1) * frobenius<2>(y2);
        y2 = raise_to_z(y3) * y0 * ret;
        return y1 * y2 * frobenius<1>(y3);
    }
}

#endif
