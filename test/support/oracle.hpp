#ifndef CUFE_TEST_ORACLE_HPP
#define CUFE_TEST_ORACLE_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include "field/field.hpp"
#include "field/tower.hpp"

#define limb_t blst_binding_limb_t
#include <blst.h>
#undef limb_t

namespace cufe::test{
    template <class T>
    struct BlstType;

    template <class Params>
    struct Oracle;

    template <>
    struct BlstType<detail::Fp>{ using type = blst_fp; };

    template <>
    struct BlstType<detail::Fr>{ using type = blst_fr; };

    template <>
    struct BlstType<detail::Fp2>{ using type = blst_fp2; };

    template <>
    struct BlstType<detail::Fp6>{ using type = blst_fp6; };

    template <>
    struct BlstType<detail::Fp12>{ using type = blst_fp12; };

    template <class T>
    using Blst = typename BlstType<T>::type;

    template <>
    struct Oracle<detail::FpParams>{
        static constexpr auto add = blst_fp_add;
        static constexpr auto sub = blst_fp_sub;
        static constexpr auto mul = blst_fp_mul;
        static constexpr auto sqr = blst_fp_sqr;
        static constexpr auto inverse = blst_fp_inverse;
        static constexpr auto from_uint64 = blst_fp_from_uint64;

        static void neg(blst_fp* result, const blst_fp* x){ blst_fp_cneg(result, x, true); }
    };

    template <>
    struct Oracle<detail::FrParams>{
        static constexpr auto add = blst_fr_add;
        static constexpr auto sub = blst_fr_sub;
        static constexpr auto mul = blst_fr_mul;
        static constexpr auto sqr = blst_fr_sqr;
        static constexpr auto inverse = blst_fr_inverse;
        static constexpr auto from_uint64 = blst_fr_from_uint64;

        static void neg(blst_fr* result, const blst_fr* x){ blst_fr_cneg(result, x, true); }
    };

    template <class Element, class... Operands>
    Element blst_result(void (*operation)(Element*, const Operands*...), const Operands&... operands){
        Element result;
        operation(&result, &operands...);
        return result;
    }

    template <class T>
    Blst<T> to_blst(const T& x){ return std::bit_cast<Blst<T>>(x); }

    template <class T>
    T from_blst(const Blst<T>& x){ return std::bit_cast<T>(x); }

    template <class Params>
    detail::Words<Params::words> blst_montgomery(const detail::Words<Params::words>& canonical){
        const auto limbs = std::bit_cast<std::array<std::uint64_t, Params::words>>(canonical);
        Blst<detail::Field<Params>> element;
        Oracle<Params>::from_uint64(&element, limbs.data());
        return std::bit_cast<detail::Words<Params::words>>(element);
    }

    inline blst_fp12 blst_conjugate(blst_fp12 x){
        blst_fp12_conjugate(&x);
        return x;
    }

    inline blst_fp12 blst_frobenius(const blst_fp12& x, const std::size_t n){
        blst_fp12 result;
        blst_fp12_frobenius_map(&result, &x, n);
        return result;
    }
}

#endif
