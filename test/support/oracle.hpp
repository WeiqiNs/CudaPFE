#ifndef CUFE_TEST_ORACLE_HPP
#define CUFE_TEST_ORACLE_HPP

#include <array>
#include <bit>
#include <cstdint>
#include "field/field.hpp"

#define limb_t blst_binding_limb_t
#include <blst.h>
#undef limb_t

namespace cufe::test{
    template <class Params>
    struct Oracle;

    template <>
    struct Oracle<detail::FpParams>{
        using Element = blst_fp;
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
        using Element = blst_fr;
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

    template <class T, class Element>
    T from_blst(const Element& element){ return std::bit_cast<T>(element); }

    template <class Params>
    typename Oracle<Params>::Element to_blst(const detail::Field<Params>& x){
        return std::bit_cast<typename Oracle<Params>::Element>(x);
    }

    template <class Params>
    detail::Words<Params::words> blst_montgomery(const detail::Words<Params::words>& canonical){
        const auto limbs = std::bit_cast<std::array<std::uint64_t, Params::words>>(canonical);
        typename Oracle<Params>::Element element;
        Oracle<Params>::from_uint64(&element, limbs.data());
        return std::bit_cast<detail::Words<Params::words>>(element);
    }
}

#endif
