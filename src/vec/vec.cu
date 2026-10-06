#include <cstddef>
#include <initializer_list>
#include <limits>
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
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/reduce.cuh"
#include "vec/reduction.hpp"
#include "vec/storage.hpp"

namespace cudapfe{
    namespace{
        using detail::Buffer;
        using detail::data;
        using detail::ElementOf;
        using detail::FieldOf;
        using detail::generate;

        template <class P>
        struct AddOp{
            const P* x;
            const P* y;
            P* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                out[i] = detail::to_affine(detail::add_mixed(detail::from_affine(x[i]), y[i]));
            }
        };

        template <class P>
        struct NegOp{
            const P* x;
            P* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = {x[i].x, -x[i].y}; }
        };

        template <class P>
        struct ScalarMulOp{
            const P* x;
            const detail::Fr* k;
            P* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                out[i] = detail::to_affine(detail::mul(detail::from_affine(x[i]), k[i]));
            }
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
            detail::Affine<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                out[i] = detail::to_affine(detail::fixed_base_mul(table, scalars[i]));
            }
        };

        CUDAPFE_HD constexpr std::size_t segment_start(
            const Spread spread, const std::size_t segment, const std::size_t length
        ){
            return spread == Spread::shared ? 0 : segment * length;
        }

        template <class F>
        struct MsmTermOp{
            const detail::Affine<F>* bases;
            const detail::Fr* scalars;
            MsmShape shape;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t t) const{
                const auto rows = shape.shape.rows;
                const auto cols = shape.shape.cols;
                const auto row = t % rows;
                const auto output = t / rows;
                const auto segment = output / cols;
                const auto base = segment_start(shape.bases, segment, rows) + row;
                const auto scalar = segment_start(shape.scalars, segment, rows * cols) + row * cols + output % cols;
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
            detail::Affine<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = detail::to_affine(in[i]); }
        };

        template <class P>
        struct PlaceOp{
            const P* in;
            std::size_t length;
            Spread spread;
            std::size_t offset;
            std::size_t width;
            P* out;

            CUDAPFE_HD void operator()(const std::size_t t) const{
                const auto segment = t / length;
                out[segment * width + offset + t % length] = in[segment_start(spread, segment, length) + t % length];
            }
        };

        std::size_t product(const std::size_t x, const std::size_t y, const char* operation){
            if (x != 0 && y > std::numeric_limits<std::size_t>::max() / x){
                throw ShapeError(std::string(operation) + " shape of " + std::to_string(x) + " x " + std::to_string(y)
                    + " overflows");
            }
            return x * y;
        }

        void require_count(const std::size_t count, const Spread spread, const std::size_t segments,
            const std::size_t length, const char* what){
            const auto expected = spread == Spread::shared ? length : product(segments, length, what);
            if (count != expected){
                throw ShapeError(std::string(what) + " needs " + std::to_string(expected) + " for its shape, got "
                    + std::to_string(count));
            }
        }

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
        using F = FieldOf<G>;
        const auto* table = engine_generator_table<E, F>();
        return generate<G, E>(scalars.size(), [&](auto* out){ return FixedBaseOp<F>{data(scalars), table, out}; });
    }

    template <class G, Engine E> requires detail::GroupPoint<G>
    Vec<G, E> msm(const Vec<G, E>& bases, const Vec<Zp, E>& scalars, const MsmShape& shape){
        using F = FieldOf<G>;
        const auto [rows, cols] = shape.shape;
        require_count(bases.size(), shape.bases, shape.segments, rows, "msm bases");
        require_count(scalars.size(), shape.scalars, shape.segments, product(rows, cols, "msm"), "msm scalars");
        const auto outputs = product(shape.segments, cols, "msm");
        if (rows == 0) return Vec<G, E>::upload(std::vector<G>(outputs));

        Buffer<detail::Jacobian<F>, E> terms(product(outputs, rows, "msm"));
        detail::for_each<E>(terms.size(), MsmTermOp<F>{data(bases), data(scalars), shape, terms.data()});
        const auto sums = detail::reduce_segments<E, detail::Jacobian<F>>(
            std::move(terms), detail::Reduction{outputs, rows}, JacobianSum{});
        return generate<G, E>(outputs, [&](auto* out){ return ToAffineOp<F>{sums.data(), out}; });
    }

    template <class G, Engine E> requires detail::GroupPoint<G>
    Vec<G, E> concat(const std::size_t segments, const std::initializer_list<Segments<G, E>> parts){
        std::size_t width = 0;
        for (const auto& part : parts){
            require_count(part.values.size(), part.spread, segments, part.length, "concat part");
            width += part.length;
        }
        Buffer<ElementOf<G>, E> out(product(segments, width, "concat"));
        std::size_t offset = 0;
        for (const auto& part : parts){
            detail::for_each<E>(segments * part.length, PlaceOp<ElementOf<G>>{
                .in = data(part.values), .length = part.length, .spread = part.spread, .offset = offset, .width = width,
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
