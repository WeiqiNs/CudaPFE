#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cufe/core.hpp>
#include <cufe/vec.hpp>
#include "curve/curve.hpp"
#include "curve/generator_table.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/storage.hpp"

namespace cufe{
    namespace{
        using detail::Buffer;
        using detail::ElementOf;

        template <class P>
        struct AddOp{
            const P* x;
            const P* y;
            P* out;

            CUFE_HD void operator()(const std::size_t i) const{
                out[i] = detail::to_affine(detail::add_mixed(detail::from_affine(x[i]), y[i]));
            }
        };

        template <class P>
        struct NegOp{
            const P* x;
            P* out;

            CUFE_HD void operator()(const std::size_t i) const{ out[i] = {x[i].x, -x[i].y}; }
        };

        template <class P>
        struct ScalarMulOp{
            const P* x;
            const detail::Fr* k;
            P* out;

            CUFE_HD void operator()(const std::size_t i) const{
                out[i] = detail::to_affine(detail::mul(detail::from_affine(x[i]), k[i]));
            }
        };

        struct GtMulOp{
            const detail::Fp12* x;
            const detail::Fp12* y;
            detail::Fp12* out;

            CUFE_HD void operator()(const std::size_t i) const{ out[i] = x[i] * y[i]; }
        };

        struct GtDivOp{
            const detail::Fp12* x;
            const detail::Fp12* y;
            detail::Fp12* out;

            CUFE_HD void operator()(const std::size_t i) const{ out[i] = x[i] * detail::conjugate(y[i]); }
        };

        template <class F>
        struct FixedBaseOp{
            const detail::Fr* scalars;
            const detail::Affine<F>* table;
            detail::Affine<F>* out;

            CUFE_HD void operator()(const std::size_t i) const{
                out[i] = detail::to_affine(detail::fixed_base_mul(table, scalars[i]));
            }
        };

        template <class F>
        const detail::DeviceArray<detail::Affine<F>>& device_generator_table(){
            static const auto table = detail::to_engine<Gpu>(detail::generator_table<F>());
            return table;
        }

        template <Engine E, class F>
        const detail::Affine<F>* engine_generator_table(){
            if constexpr (std::same_as<E, Cpu>) return detail::generator_table<F>().data();
            else return device_generator_table<F>().data();
        }

        template <class T, Engine E>
        const ElementOf<T>* data(const Vec<T, E>& values){ return detail::buffer(values).data(); }

        template <class T, Engine E, class MakeOp>
        Vec<T, E> generate(const std::size_t size, const MakeOp& make_op){
            Buffer<ElementOf<T>, E> out(size);
            detail::for_each<E>(size, make_op(out.data()));
            detail::finish<E>();
            return detail::vec<T, E>(std::move(out));
        }
    }

    template <class T, Engine E>
    Vec<T, E>::Vec() : Vec(detail::vec<T, E>(Buffer<ElementOf<T>, E>(0))){}

    template <class T, Engine E>
    Vec<T, E>::Vec(std::shared_ptr<const detail::Storage<T, E>> storage) : storage_(std::move(storage)){}

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::upload(const std::span<const T> values){
        return detail::vec<T, E>(detail::to_engine<E>(detail::to_elements(values)));
    }

    template <class T, Engine E>
    std::vector<T> Vec<T, E>::download() const{
        return detail::from_elements<T>(detail::to_host(storage_->buffer));
    }

    template <class T, Engine E>
    T Vec<T, E>::at(const std::size_t index) const{
        if (index >= size()){
            throw std::out_of_range("Vec::at(" + std::to_string(index) + ") on a vector of " + std::to_string(size()));
        }
        return detail::from_element<T>(detail::element_at(storage_->buffer, index));
    }

    template <class T, Engine E>
    std::size_t Vec<T, E>::size() const{
        return storage_->buffer.size();
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::plus(const Vec& y) const requires detail::GroupPoint<T>{
        detail::require_same_size(*this, y, "point vector addition");
        return generate<T, E>(size(), [&](auto* out){ return AddOp<ElementOf<T>>{data(*this), data(y), out}; });
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::negated() const requires detail::GroupPoint<T>{
        return generate<T, E>(size(), [&](auto* out){ return NegOp<ElementOf<T>>{data(*this), out}; });
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::scaled(const Vec<Zp, E>& k) const requires detail::GroupPoint<T>{
        detail::require_same_size(*this, k, "point vector scaling");
        return generate<T, E>(size(), [&](auto* out){ return ScalarMulOp<ElementOf<T>>{data(*this), data(k), out}; });
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::times(const Vec& y) const requires std::same_as<T, Gt>{
        detail::require_same_size(*this, y, "Gt vector product");
        return generate<T, E>(size(), [&](auto* out){ return GtMulOp{data(*this), data(y), out}; });
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::divided(const Vec& y) const requires std::same_as<T, Gt>{
        detail::require_same_size(*this, y, "Gt vector quotient");
        return generate<T, E>(size(), [&](auto* out){ return GtDivOp{data(*this), data(y), out}; });
    }

    template <class G, Engine E> requires detail::GroupPoint<G>
    Vec<G, E> mul_generator(const Vec<Zp, E>& scalars){
        using F = decltype(ElementOf<G>::x);
        const auto* table = engine_generator_table<E, F>();
        return generate<G, E>(scalars.size(), [&](auto* out){ return FixedBaseOp<F>{data(scalars), table, out}; });
    }

    template class Vec<Zp, Cpu>;
    template class Vec<Zp, Gpu>;
    template class Vec<G1, Cpu>;
    template class Vec<G1, Gpu>;
    template class Vec<G2, Cpu>;
    template class Vec<G2, Gpu>;
    template class Vec<Gt, Cpu>;
    template class Vec<Gt, Gpu>;

    template Vec<G1, Cpu> mul_generator<G1, Cpu>(const Vec<Zp, Cpu>&);
    template Vec<G1, Gpu> mul_generator<G1, Gpu>(const Vec<Zp, Gpu>&);
    template Vec<G2, Cpu> mul_generator<G2, Cpu>(const Vec<Zp, Cpu>&);
    template Vec<G2, Gpu> mul_generator<G2, Gpu>(const Vec<Zp, Gpu>&);
}
