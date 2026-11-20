#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cudapfe/core.hpp>
#include <cudapfe/vec.hpp>
#include "curve/curve.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/placement.hpp"
#include "vec/points.cuh"
#include "vec/reduce.cuh"
#include "vec/reduction.hpp"
#include "vec/spread.hpp"
#include "vec/storage.hpp"

namespace cudapfe{
    namespace{
        using detail::Buffer;
        using detail::data;
        using detail::ElementOf;
        using detail::FieldOf;
        using detail::generate;
        using detail::generate_points;

        template <class F>
        struct AddOp{
            const detail::Affine<F>* x;
            const detail::Affine<F>* y;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                out[i] = detail::add_mixed(detail::from_affine(x[i]), y[i]);
            }
        };

        template <class P>
        struct NegOp{
            const P* x;
            P* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = {x[i].x, -x[i].y}; }
        };

        struct GtMulOp{
            const detail::Fp12* x;
            const detail::Fp12* y;
            detail::Fp12* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = x[i] * y[i]; }
        };

        struct GtDivOp{
            const detail::Fp12* x;
            const detail::Fp12* y;
            detail::Fp12* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = x[i] * detail::conjugate(y[i]); }
        };

        template <class P>
        struct PlaceOp{
            const P* in;
            detail::Layout layout;
            std::size_t offset;
            std::size_t width;
            P* out;

            CUDAPFE_HD void operator()(const std::size_t t) const{
                const auto segment = t / layout.length;
                const auto k = t % layout.length;
                out[segment * width + offset + k] = in[layout.offset(segment) + k];
            }
        };

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
    Vec<T, detail::OtherEngine<E>> Vec<T, E>::transferred() const{
        using To = detail::OtherEngine<E>;
        return detail::vec<T, To>(detail::to_engine<To>(detail::to_host(storage_->buffer)));
    }

    template <class T, Engine E>
    std::size_t Vec<T, E>::size() const{
        return storage_->buffer.size();
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::plus(const Vec& y) const requires detail::GroupPoint<T>{
        detail::require_same_size(*this, y, "point vector addition");
        if constexpr (std::same_as<E, Gpu>){
            if (detail::points_on_host(size())) return (to<Cpu>() + y.template to<Cpu>()).template to<Gpu>();
        }
        return generate_points<T, E>(size(), [&](auto* out){ return AddOp<FieldOf<T>>{data(*this), data(y), out}; });
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::negated() const requires detail::GroupPoint<T>{
        return generate<T, E>(size(), [&](auto* out){ return NegOp<ElementOf<T>>{data(*this), out}; });
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::scaled(const Vec<Zp, E>& k) const requires detail::GroupPoint<T>{
        detail::require_same_size(*this, k, "point vector scaling");
        if constexpr (std::same_as<E, Gpu>){
            if (detail::points_on_host(size())) return (to<Cpu>() * k.template to<Cpu>()).template to<Gpu>();
        }
        return detail::scale_points(*this, k);
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
    Vec<G, E> concat(const std::size_t segments, const std::initializer_list<Segments<G, E>> parts){
        std::size_t width = 0;
        for (const auto& part : parts){
            detail::Layout{segments, part.length, part.spread}.require_size(part.values.size(), "concat part");
            width += part.length;
        }
        Buffer<ElementOf<G>, E> out(detail::checked_product(segments, width, "concat"));
        std::size_t offset = 0;
        for (const auto& part : parts){
            detail::for_each<E>(segments * part.length, PlaceOp<ElementOf<G>>{
                .in = data(part.values), .layout = {segments, part.length, part.spread}, .offset = offset, .width = width,
                .out = out.data()
            });
            offset += part.length;
        }
        detail::finish<E>();
        return detail::vec<G, E>(std::move(out));
    }

    template class Vec<Zp, Cpu>;
    template class Vec<Zp, Gpu>;
    template class Vec<G1, Cpu>;
    template class Vec<G1, Gpu>;
    template class Vec<G2, Cpu>;
    template class Vec<G2, Gpu>;
    template class Vec<Gt, Cpu>;
    template class Vec<Gt, Gpu>;

    template Vec<G1, Cpu> concat<G1, Cpu>(std::size_t, std::initializer_list<Segments<G1, Cpu>>);
    template Vec<G1, Gpu> concat<G1, Gpu>(std::size_t, std::initializer_list<Segments<G1, Gpu>>);
    template Vec<G2, Cpu> concat<G2, Cpu>(std::size_t, std::initializer_list<Segments<G2, Cpu>>);
    template Vec<G2, Gpu> concat<G2, Gpu>(std::size_t, std::initializer_list<Segments<G2, Gpu>>);
}
