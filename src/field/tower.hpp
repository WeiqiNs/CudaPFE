#ifndef CUDAPFE_FIELD_TOWER_HPP
#define CUDAPFE_FIELD_TOWER_HPP

#include <type_traits>
#include "field/constants.hpp"
#include "field/field.hpp"
#include "support/hd.hpp"

namespace cudapfe::detail{
    struct Fp2{
        Fp c0, c1;

        [[nodiscard]] CUDAPFE_HD static Fp2 zero(){ return {Fp::zero(), Fp::zero()}; }

        [[nodiscard]] CUDAPFE_HD static Fp2 one(){ return {Fp::one(), Fp::zero()}; }

        [[nodiscard]] CUDAPFE_HD static Fp2 from_montgomery(const Fp2Words& words){
            return {Fp::from_montgomery(words[0]), Fp::from_montgomery(words[1])};
        }

        [[nodiscard]] CUDAPFE_HD bool is_zero() const{ return c0.is_zero() && c1.is_zero(); }

        [[nodiscard]] CUDAPFE_HD Fp2 square() const{
            const auto t = c0 * c1;
            return {(c0 + c1) * (c0 - c1), t + t};
        }

        [[nodiscard]] CUDAPFE_HD Fp2 inverse() const{
            const auto t = (c0.square() + c1.square()).inverse();
            return {c0 * t, -(c1 * t)};
        }

        friend CUDAPFE_HD Fp2 operator+(const Fp2& x, const Fp2& y){ return {x.c0 + y.c0, x.c1 + y.c1}; }

        friend CUDAPFE_HD Fp2 operator-(const Fp2& x, const Fp2& y){ return {x.c0 - y.c0, x.c1 - y.c1}; }

        friend CUDAPFE_HD Fp2 operator-(const Fp2& x){ return {-x.c0, -x.c1}; }

        friend CUDAPFE_HD Fp2 operator*(const Fp2& x, const Fp2& y){
            const auto t0 = x.c0 * y.c0;
            const auto t1 = x.c1 * y.c1;
            return {t0 - t1, (x.c0 + x.c1) * (y.c0 + y.c1) - t0 - t1};
        }

        friend CUDAPFE_HD Fp2 operator*(const Fp2& x, const Fp& s){ return {x.c0 * s, x.c1 * s}; }

        friend CUDAPFE_HD bool operator==(const Fp2& x, const Fp2& y){ return x.c0 == y.c0 && x.c1 == y.c1; }
    };

    [[nodiscard]] CUDAPFE_HD Fp2 mul_by_nonresidue(const Fp2& x){ return {x.c0 - x.c1, x.c0 + x.c1}; }

    struct Fp4{
        Fp2 c0, c1;
    };

    struct Fp6{
        Fp2 c0, c1, c2;

        [[nodiscard]] CUDAPFE_HD static Fp6 zero(){ return {Fp2::zero(), Fp2::zero(), Fp2::zero()}; }

        [[nodiscard]] CUDAPFE_HD static Fp6 one(){ return {Fp2::one(), Fp2::zero(), Fp2::zero()}; }

        [[nodiscard]] CUDAPFE_HD_CALL Fp6 square() const{
            const auto s0 = c0.square();
            const auto p01 = c0 * c1;
            const auto p12 = c1 * c2;
            const auto m01 = p01 + p01;
            const auto m12 = p12 + p12;
            const auto s2 = c2.square();
            return {
                mul_by_nonresidue(m12) + s0,
                mul_by_nonresidue(s2) + m01,
                (c2 + c1 + c0).square() - s0 - s2 - m01 - m12
            };
        }

        [[nodiscard]] CUDAPFE_HD Fp6 inverse() const{
            const auto t0 = c0.square() - mul_by_nonresidue(c1 * c2);
            const auto t1 = mul_by_nonresidue(c2.square()) - c0 * c1;
            const auto t2 = c1.square() - c0 * c2;
            const auto t = (mul_by_nonresidue(t1 * c2 + t2 * c1) + t0 * c0).inverse();
            return {t0 * t, t1 * t, t2 * t};
        }

        friend CUDAPFE_HD Fp6 operator+(const Fp6& x, const Fp6& y){ return {x.c0 + y.c0, x.c1 + y.c1, x.c2 + y.c2}; }

        friend CUDAPFE_HD Fp6 operator-(const Fp6& x, const Fp6& y){ return {x.c0 - y.c0, x.c1 - y.c1, x.c2 - y.c2}; }

        friend CUDAPFE_HD Fp6 operator-(const Fp6& x){ return {-x.c0, -x.c1, -x.c2}; }

        friend CUDAPFE_HD_CALL Fp6 operator*(const Fp6& x, const Fp6& y){
            const auto t0 = x.c0 * y.c0;
            const auto t1 = x.c1 * y.c1;
            const auto t2 = x.c2 * y.c2;
            return {
                mul_by_nonresidue((x.c1 + x.c2) * (y.c1 + y.c2) - t1 - t2) + t0,
                (x.c0 + x.c1) * (y.c0 + y.c1) - t0 - t1 + mul_by_nonresidue(t2),
                (x.c0 + x.c2) * (y.c0 + y.c2) - t0 - t2 + t1
            };
        }

        friend CUDAPFE_HD bool operator==(const Fp6& x, const Fp6& y){
            return x.c0 == y.c0 && x.c1 == y.c1 && x.c2 == y.c2;
        }
    };

    [[nodiscard]] CUDAPFE_HD Fp6 mul_by_v(const Fp6& x){ return {mul_by_nonresidue(x.c2), x.c0, x.c1}; }

    struct Fp12{
        Fp6 c0, c1;

        [[nodiscard]] CUDAPFE_HD static Fp12 one(){ return {Fp6::one(), Fp6::zero()}; }

        [[nodiscard]] CUDAPFE_HD_CALL Fp12 square() const{
            const auto t0 = (c0 + c1) * (c0 + mul_by_v(c1));
            const auto t1 = c0 * c1;
            return {t0 - t1 - mul_by_v(t1), t1 + t1};
        }

        [[nodiscard]] CUDAPFE_HD Fp12 inverse() const{
            const auto t = (c0.square() - mul_by_v(c1.square())).inverse();
            return {c0 * t, -(c1 * t)};
        }

        friend CUDAPFE_HD_CALL Fp12 operator*(const Fp12& x, const Fp12& y){
            const auto t0 = x.c0 * y.c0;
            const auto t1 = x.c1 * y.c1;
            return {t0 + mul_by_v(t1), (x.c0 + x.c1) * (y.c0 + y.c1) - t0 - t1};
        }

        friend CUDAPFE_HD bool operator==(const Fp12& x, const Fp12& y){ return x.c0 == y.c0 && x.c1 == y.c1; }
    };

    static_assert(sizeof(Fp2) == 96 && std::is_trivially_copyable_v<Fp2>);
    static_assert(sizeof(Fp6) == 288 && std::is_trivially_copyable_v<Fp6>);
    static_assert(sizeof(Fp12) == 576 && std::is_trivially_copyable_v<Fp12>);

    [[nodiscard]] CUDAPFE_HD Fp2 conjugate(const Fp2& x){ return {x.c0, -x.c1}; }

    [[nodiscard]] CUDAPFE_HD Fp12 conjugate(const Fp12& x){ return {x.c0, -x.c1}; }

    [[nodiscard]] CUDAPFE_HD Fp6 mul_by_xy0(const Fp6& x, const Fp2& y0, const Fp2& y1){
        const auto t0 = x.c0 * y0;
        const auto t1 = x.c1 * y1;
        return {mul_by_nonresidue(x.c2 * y1) + t0, (x.c0 + x.c1) * (y0 + y1) - t0 - t1, x.c2 * y0 + t1};
    }

    [[nodiscard]] CUDAPFE_HD Fp6 mul_by_0y0(const Fp6& x, const Fp2& y){
        return {mul_by_nonresidue(x.c2 * y), x.c0 * y, x.c1 * y};
    }

    [[nodiscard]] CUDAPFE_HD_CALL Fp12 mul_by_line(const Fp12& x, const Fp6& xy00z0){
        const auto t0 = mul_by_xy0(x.c0, xy00z0.c0, xy00z0.c1);
        const auto t1 = mul_by_0y0(x.c1, xy00z0.c2);
        return {t0 + mul_by_v(t1), mul_by_xy0(x.c0 + x.c1, xy00z0.c0, xy00z0.c1 + xy00z0.c2) - t0 - t1};
    }

    [[nodiscard]] CUDAPFE_HD Fp4 square_fp4(const Fp2& x0, const Fp2& x1){
        const auto t0 = x0.square();
        const auto t1 = x1.square();
        return {mul_by_nonresidue(t1) + t0, (x0 + x1).square() - t0 - t1};
    }

    [[nodiscard]] CUDAPFE_HD Fp2 thrice_minus_twice(const Fp2& square, const Fp2& x){
        const auto t = square - x;
        return t + t + square;
    }

    [[nodiscard]] CUDAPFE_HD Fp2 thrice_plus_twice(const Fp2& square, const Fp2& x){
        const auto t = square + x;
        return t + t + square;
    }

    [[nodiscard]] CUDAPFE_HD_CALL Fp12 cyclotomic_square(const Fp12& x){
        const auto t0 = square_fp4(x.c0.c0, x.c1.c1);
        const auto t1 = square_fp4(x.c1.c0, x.c0.c2);
        const auto t2 = square_fp4(x.c0.c1, x.c1.c2);
        return {
            {
                thrice_minus_twice(t0.c0, x.c0.c0),
                thrice_minus_twice(t1.c0, x.c0.c1),
                thrice_minus_twice(t2.c0, x.c0.c2)
            },
            {
                thrice_plus_twice(mul_by_nonresidue(t2.c1), x.c1.c0),
                thrice_plus_twice(t0.c1, x.c1.c1),
                thrice_plus_twice(t1.c1, x.c1.c2)
            }
        };
    }

    template <int N>
    [[nodiscard]] CUDAPFE_HD Fp2 frobenius(const Fp2& x){
        if constexpr (N % 2 == 1) return conjugate(x);
        else return x;
    }

    template <int N>
    [[nodiscard]] CUDAPFE_HD Fp6 frobenius(const Fp6& x){
        constexpr auto c1 = kFrobeniusFp6C1[N - 1];
        constexpr auto c2 = kFrobeniusFp6C2[N - 1];
        return {
            frobenius<N>(x.c0), frobenius<N>(x.c1) * Fp2::from_montgomery(c1),
            frobenius<N>(x.c2) * Fp::from_montgomery(c2)
        };
    }

    template <int N>
    [[nodiscard]] CUDAPFE_HD Fp12 frobenius(const Fp12& x){
        static_assert(N >= 1 && N <= 3);
        constexpr auto words = kFrobeniusFp12[N - 1];
        const auto c = Fp2::from_montgomery(words);
        const auto c1 = frobenius<N>(x.c1);
        return {frobenius<N>(x.c0), {c1.c0 * c, c1.c1 * c, c1.c2 * c}};
    }
}

#endif
