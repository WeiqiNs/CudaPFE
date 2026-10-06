#ifndef CUDAPFE_TEST_ORACLE_HPP
#define CUDAPFE_TEST_ORACLE_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/core.hpp>
#include <support/samples.hpp>
#include "curve/curve.hpp"
#include "curve/encoding.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"

#define limb_t blst_binding_limb_t
#include <blst.h>
#undef limb_t

namespace cudapfe::test{
    template <class T>
    struct BlstType;

    template <class Params>
    struct Oracle;

    template <class F>
    struct CurveOracle;

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

    template <>
    struct BlstType<detail::G1Jacobian>{ using type = blst_p1; };

    template <>
    struct BlstType<detail::G1Affine>{ using type = blst_p1_affine; };

    template <>
    struct BlstType<detail::G2Jacobian>{ using type = blst_p2; };

    template <>
    struct BlstType<detail::G2Affine>{ using type = blst_p2_affine; };

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

    template <>
    struct CurveOracle<detail::Fp>{
        static constexpr auto add = blst_p1_add_or_double;
        static constexpr auto add_affine = blst_p1_add_or_double_affine;
        static constexpr auto dbl = blst_p1_double;
        static constexpr auto to_affine = blst_p1_to_affine;
        static constexpr auto generator = blst_p1_generator;
        static constexpr auto mult = blst_p1_mult;
        static constexpr auto cneg = blst_p1_cneg;
        static constexpr auto compress = blst_p1_affine_compress;
        static constexpr auto serialize = blst_p1_affine_serialize;
        static constexpr auto uncompress = blst_p1_uncompress;
        static constexpr auto deserialize = blst_p1_deserialize;
        static constexpr auto in_group = blst_p1_affine_in_g1;
    };

    template <>
    struct CurveOracle<detail::Fp2>{
        static constexpr auto add = blst_p2_add_or_double;
        static constexpr auto add_affine = blst_p2_add_or_double_affine;
        static constexpr auto dbl = blst_p2_double;
        static constexpr auto to_affine = blst_p2_to_affine;
        static constexpr auto generator = blst_p2_generator;
        static constexpr auto mult = blst_p2_mult;
        static constexpr auto cneg = blst_p2_cneg;
        static constexpr auto compress = blst_p2_affine_compress;
        static constexpr auto serialize = blst_p2_affine_serialize;
        static constexpr auto uncompress = blst_p2_uncompress;
        static constexpr auto deserialize = blst_p2_deserialize;
        static constexpr auto in_group = blst_p2_affine_in_g2;
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

    class SideNames{
    public:
        template <class F>
        static std::string GetName(int){ return std::string(detail::kGroupName<F>); }
    };

    using Sides = ::testing::Types<detail::Fp, detail::Fp2>;

    inline std::array<std::uint8_t, 32> le_bytes(const detail::Fr& k){
        return std::bit_cast<std::array<std::uint8_t, 32>>(k.canonical());
    }

    template <class F>
    detail::Affine<F> blst_affine(const Blst<detail::Jacobian<F>>& p){
        return from_blst<detail::Affine<F>>(blst_result(CurveOracle<F>::to_affine, p));
    }

    template <class F>
    Blst<detail::Jacobian<F>> blst_mult(const Blst<detail::Jacobian<F>>& p, const detail::Fr& k){
        Blst<detail::Jacobian<F>> product;
        CurveOracle<F>::mult(&product, &p, le_bytes(k).data(), 255);
        return product;
    }

    template <class F>
    std::vector<detail::Jacobian<F>> random_points(const std::size_t count){
        std::vector<detail::Jacobian<F>> points;
        for (const auto& k : field_samples<detail::FrParams>(count)){
            points.push_back(from_blst<detail::Jacobian<F>>(blst_mult<F>(*CurveOracle<F>::generator(), k)));
        }
        return points;
    }

    template <class F>
    Bytes small_x_compressed(const std::uint8_t x){
        Bytes bytes(detail::encoded_size<F>(Encoding::compressed), 0);
        bytes.front() = detail::kCompressedFlag;
        bytes.back() = x;
        return bytes;
    }

    template <class F>
    detail::Affine<F> first_point_outside_subgroup(){
        using O = CurveOracle<F>;
        for (std::uint8_t x = 1;; ++x){
            Blst<detail::Affine<F>> point;
            if (O::uncompress(&point, small_x_compressed<F>(x).data()) == BLST_SUCCESS && !O::in_group(&point)){
                return from_blst<detail::Affine<F>>(point);
            }
        }
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
