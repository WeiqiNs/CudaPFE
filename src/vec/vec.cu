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
#include "curve/generator_table.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
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

        constexpr std::size_t kAffineChunk = 8;

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

        template <class F>
        struct ScalarMulOp{
            const detail::Affine<F>* x;
            const detail::Fr* k;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = detail::mul(detail::from_affine(x[i]), k[i]); }
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

        template <class F>
        struct FixedBaseOp{
            const detail::Fr* scalars;
            const detail::Affine<F>* table;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = detail::fixed_base_mul(table, scalars[i]); }
        };

        template <class F>
        struct MsmTermOp{
            const detail::Affine<F>* bases;
            const detail::Fr* scalars;
            detail::Layout base_layout;
            detail::Layout scalar_layout;
            std::size_t cols;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t t) const{
                const auto rows = base_layout.length;
                const auto row = t % rows;
                const auto output = t / rows;
                const auto segment = output / cols;
                const auto base = base_layout.offset(segment) + row;
                const auto scalar = scalar_layout.offset(segment) + row * cols + output % cols;
                out[t] = detail::mul(detail::from_affine(bases[base]), scalars[scalar]);
            }
        };

        struct JacobianSum{
            template <class F>
            CUDAPFE_HD detail::Jacobian<F> operator()(const detail::Jacobian<F>& p, const detail::Jacobian<F>& q) const{
                return detail::add(p, q);
            }
        };

        template <class F>
        struct ToAffineOp{
            const detail::Jacobian<F>* in;
            detail::Chunks chunks;
            detail::Affine<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                const auto chunk = chunks.at(i);
                detail::to_affine(in + chunk.first, chunk.count, out + chunk.first);
            }
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

        template <class G, Engine E>
        Vec<G, E> normalized(const Buffer<detail::Jacobian<FieldOf<G>>, E>& points){
            Buffer<ElementOf<G>, E> out(points.size());
            const detail::Chunks chunks{1, points.size(), kAffineChunk};
            detail::for_each<E>(chunks.count(), ToAffineOp<FieldOf<G>>{points.data(), chunks, out.data()});
            detail::finish<E>();
            return detail::vec<G, E>(std::move(out));
        }

        template <class G, Engine E, class MakeOp>
        Vec<G, E> generate_points(const std::size_t size, const MakeOp& make_op){
            Buffer<detail::Jacobian<FieldOf<G>>, E> points(size);
            detail::for_each<E>(size, make_op(points.data()));
            return normalized<G, E>(points);
        }

        template <Engine E, class F>
        const detail::Affine<F>* engine_generator_table(){
            if constexpr (std::same_as<E, Cpu>){
                return detail::generator_table<F>().data();
            } else {
                static const auto table = detail::to_engine<Gpu>(detail::generator_table<F>());
                return table.data();
            }
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
        return generate_points<T, E>(size(), [&](auto* out){ return AddOp<FieldOf<T>>{data(*this), data(y), out}; });
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::negated() const requires detail::GroupPoint<T>{
        return generate<T, E>(size(), [&](auto* out){ return NegOp<ElementOf<T>>{data(*this), out}; });
    }

    template <class T, Engine E>
    Vec<T, E> Vec<T, E>::scaled(const Vec<Zp, E>& k) const requires detail::GroupPoint<T>{
        detail::require_same_size(*this, k, "point vector scaling");
        return generate_points<T, E>(size(), [&](auto* out){ return ScalarMulOp<FieldOf<T>>{data(*this), data(k), out}; });
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
        using F = FieldOf<G>;
        const auto* table = engine_generator_table<E, F>();
        return generate_points<G, E>(scalars.size(), [&](auto* out){ return FixedBaseOp<F>{data(scalars), table, out}; });
    }

    template <class G, Engine E> requires detail::GroupPoint<G>
    Vec<G, E> msm(const Vec<G, E>& bases, const Vec<Zp, E>& scalars, const MsmShape& shape){
        using F = FieldOf<G>;
        const auto [rows, cols] = shape.shape;
        const detail::Layout base_layout{shape.segments, rows, shape.bases};
        base_layout.require_size(bases.size(), "msm bases");
        const detail::Layout scalar_layout{shape.segments, detail::checked_product(rows, cols, "msm"), shape.scalars};
        scalar_layout.require_size(scalars.size(), "msm scalars");
        const auto outputs = detail::checked_product(shape.segments, cols, "msm");
        if (rows == 0) return Vec<G, E>::upload(std::vector<G>(outputs));

        Buffer<detail::Jacobian<F>, E> terms(detail::checked_product(outputs, rows, "msm"));
        detail::for_each<E>(terms.size(), MsmTermOp<F>{
            data(bases), data(scalars), base_layout, scalar_layout, cols, terms.data()
        });
        const auto sums = detail::reduce_segments<E, detail::Jacobian<F>>(
            std::move(terms), {outputs, rows, detail::kReduceFanIn}, JacobianSum{});
        return normalized<G, E>(sums);
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

    template Vec<G1, Cpu> mul_generator<G1, Cpu>(const Vec<Zp, Cpu>&);
    template Vec<G1, Gpu> mul_generator<G1, Gpu>(const Vec<Zp, Gpu>&);
    template Vec<G2, Cpu> mul_generator<G2, Cpu>(const Vec<Zp, Cpu>&);
    template Vec<G2, Gpu> mul_generator<G2, Gpu>(const Vec<Zp, Gpu>&);
    template Vec<G1, Cpu> msm<G1, Cpu>(const Vec<G1, Cpu>&, const Vec<Zp, Cpu>&, const MsmShape&);
    template Vec<G1, Gpu> msm<G1, Gpu>(const Vec<G1, Gpu>&, const Vec<Zp, Gpu>&, const MsmShape&);
    template Vec<G2, Cpu> msm<G2, Cpu>(const Vec<G2, Cpu>&, const Vec<Zp, Cpu>&, const MsmShape&);
    template Vec<G2, Gpu> msm<G2, Gpu>(const Vec<G2, Gpu>&, const Vec<Zp, Gpu>&, const MsmShape&);
    template Vec<G1, Cpu> concat<G1, Cpu>(std::size_t, std::initializer_list<Segments<G1, Cpu>>);
    template Vec<G1, Gpu> concat<G1, Gpu>(std::size_t, std::initializer_list<Segments<G1, Gpu>>);
    template Vec<G2, Cpu> concat<G2, Cpu>(std::size_t, std::initializer_list<Segments<G2, Cpu>>);
    template Vec<G2, Gpu> concat<G2, Gpu>(std::size_t, std::initializer_list<Segments<G2, Gpu>>);
}
