#ifndef CUDAPFE_VEC_MSM_CUH
#define CUDAPFE_VEC_MSM_CUH

// msm's definition, instantiated per group by msm_g1.cu and msm_g2.cu so the two compile in parallel.
#include <cstddef>
#include <vector>
#include <cudapfe/core.hpp>
#include <cudapfe/vec.hpp>
#include "curve/curve.hpp"
#include "curve/straus.hpp"
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
        using detail::FieldOf;
        using detail::JacobianSum;

        struct MsmLayout{
            detail::Layout bases;
            detail::Layout scalars;
            std::size_t cols;

            [[nodiscard]] CUDAPFE_HD std::size_t rows() const{ return bases.length; }

            [[nodiscard]] CUDAPFE_HD std::size_t outputs() const{ return bases.segments * cols; }

            [[nodiscard]] CUDAPFE_HD std::size_t base(const std::size_t output, const std::size_t row) const{
                return bases.offset(output / cols) + row;
            }

            [[nodiscard]] CUDAPFE_HD std::size_t scalar(const std::size_t output, const std::size_t row) const{
                return scalars.offset(output / cols) + row * cols + output % cols;
            }
        };

        template <class F>
        struct MsmTermOp{
            enum : std::size_t{ host_parallel_from = 2 };

            MsmLayout layout;
            const detail::Affine<F>* bases;
            const detail::Fr* scalars;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t t) const{
                const auto rows = layout.rows();
                const auto row = t % rows;
                const auto output = t / rows;
                out[t] = detail::mul(detail::from_affine(bases[layout.base(output, row)]),
                    scalars[layout.scalar(output, row)]);
            }
        };

        struct SignedScalarOp{
            const detail::Fr* in;
            detail::SignedScalar* out;
            std::size_t* bits;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                out[i] = detail::signed_scalar(in[i]);
                bits[i] = detail::bit_length(out[i].magnitude);
            }
        };

        struct MaxBitLength{
            CUDAPFE_HD std::size_t operator()(const std::size_t x, const std::size_t y) const{ return x < y ? y : x; }
        };

        template <Engine E>
        struct SignedScalars{
            Buffer<detail::SignedScalar, E> values;
            std::size_t bits;
        };

        template <class F>
        struct MsmStrausOp{
            enum : std::size_t{ host_parallel_from = 2 };

            detail::Chunks items;
            MsmLayout layout;
            std::size_t bits;
            const detail::Affine<F>* bases;
            const detail::SignedScalar* scalars;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                const auto item = items.at(i);
                out[i] = detail::straus_sum(detail::MsmTerms<F>{
                    .bases = bases + layout.base(item.segment, item.first),
                    .scalars = scalars + layout.scalar(item.segment, item.first),
                    .scalar_stride = layout.cols,
                    .count = item.count,
                    .bits = bits,
                });
            }
        };

        template <Engine E>
        SignedScalars<E> signed_scalars(const Vec<Zp, E>& scalars){
            const auto size = scalars.size();
            Buffer<detail::SignedScalar, E> values(size);
            Buffer<std::size_t, E> bits(size);
            detail::for_each<E>(size, SignedScalarOp{data(scalars), values.data(), bits.data()});
            const auto longest = detail::reduce_segments<E, std::size_t>(
                std::move(bits), {1, size, detail::kReduceFanIn}, MaxBitLength{});
            return {std::move(values), detail::element_at(longest, 0)};
        }

        template <class G, Engine E>
        Buffer<detail::Jacobian<FieldOf<G>>, E> term_sums(
            const Vec<G, E>& bases, const Vec<Zp, E>& scalars, const MsmLayout& layout
        ){
            using F = FieldOf<G>;
            const auto outputs = layout.outputs();
            const auto rows = layout.rows();
            Buffer<detail::Jacobian<F>, E> terms(outputs * rows);
            detail::for_each<E>(terms.size(), MsmTermOp<F>{layout, data(bases), data(scalars), terms.data()});
            return detail::reduce_segments<E, detail::Jacobian<F>>(
                std::move(terms), {outputs, rows, detail::kReduceFanIn}, JacobianSum{});
        }

        template <class G, Engine E>
        Buffer<detail::Jacobian<FieldOf<G>>, E> straus_sums(
            const Vec<G, E>& bases, const SignedScalars<E>& scalars, const MsmLayout& layout
        ){
            using F = FieldOf<G>;
            const auto outputs = layout.outputs();
            const auto rows = layout.rows();
            const auto threads = detail::resident_threads<E, MsmStrausOp<F>>();
            const detail::Chunks items{outputs, rows, detail::chunk_size(outputs * rows, threads, rows)};
            Buffer<detail::Jacobian<F>, E> partials(items.count());
            detail::for_each<E>(items.count(), MsmStrausOp<F>{
                items, layout, scalars.bits, data(bases), scalars.values.data(), partials.data()
            });
            return detail::reduce_segments<E, detail::Jacobian<F>>(
                std::move(partials), {outputs, items.per_segment(), detail::kReduceFanIn}, JacobianSum{});
        }
    }

    template <class G, Engine E> requires detail::GroupPoint<G>
    Vec<G, E> msm(const Vec<G, E>& bases, const Vec<Zp, E>& scalars, const MsmShape& shape){
        const auto [rows, cols] = shape.shape;
        const detail::Layout base_layout{shape.segments, rows, shape.bases};
        base_layout.require_size(bases.size(), "msm bases");
        const detail::Layout scalar_layout{shape.segments, detail::checked_product(rows, cols, "msm"), shape.scalars};
        scalar_layout.require_size(scalars.size(), "msm scalars");
        const auto outputs = detail::checked_product(shape.segments, cols, "msm");
        const auto count = detail::checked_product(outputs, rows, "msm");
        if (count == 0) return Vec<G, E>::upload(std::vector<G>(outputs));
        if constexpr (std::same_as<E, Gpu>){
            if (detail::points_on_host(count)){
                return msm(bases.template to<Cpu>(), scalars.template to<Cpu>(), shape).template to<Gpu>();
            }
        }

        const MsmLayout layout{base_layout, scalar_layout, cols};
        if (const auto signed_values = signed_scalars(scalars); signed_values.bits <= detail::kStrausMaxBits){
            return detail::normalized<G, E>(straus_sums(bases, signed_values, layout));
        }
        return detail::normalized<G, E>(term_sums(bases, scalars, layout));
    }
}

#endif
