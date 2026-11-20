#include <algorithm>
#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>
#include <cudapfe/core.hpp>
#include <cudapfe/pairing.hpp>
#include <cudapfe/vec.hpp>
#include "curve/curve.hpp"
#include "field/tower.hpp"
#include "pairing/miller.hpp"
#include "pairing/plan.hpp"
#include "support/access.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/reduce.cuh"
#include "vec/reduction.hpp"
#include "vec/spread.hpp"
#include "vec/storage.hpp"

namespace cudapfe{
    namespace detail{
        template <Engine E>
        struct LineStorage{
            Vec<G2, E> sources;
            Buffer<Line, E> lines;
        };
    }

    namespace{
        using detail::Access;
        using detail::Buffer;
        using detail::data;
        using detail::Fp12;
        using detail::LineStorage;

        struct PrepareLinesOp{
            enum : std::size_t{ host_parallel_from = 2 };

            const detail::G2Affine* qs;
            detail::Line* lines;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                detail::prepare_lines(qs[i], lines + i * detail::kLineCount);
            }
        };

        static_assert(detail::kMaxPairsPerItem <= detail::MaskedPairs::capacity);

        struct MillerItemOp{
            enum : std::size_t{ host_parallel_from = 2 };

            detail::Chunks items;
            detail::Layout p_layout;
            detail::Layout q_layout;
            const detail::G1Affine* ps;
            const detail::G2Affine* qs;
            const detail::Line* lines;
            Fp12* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                const auto item = items.at(i);
                const auto p = p_layout.offset(item.segment) + item.first;
                const auto q = q_layout.offset(item.segment) + item.first;
                const detail::PreparedPairs pairs{ps + p, qs + q, lines + q * detail::kLineCount, item.count};
#ifdef __CUDA_ARCH__
                out[i] = detail::miller(detail::MaskedPairs::gather(pairs));
#else
                out[i] = detail::miller(pairs);
#endif
            }
        };

        struct Fp12Product{
            CUDAPFE_HD Fp12 operator()(const Fp12& x, const Fp12& y) const{ return x * y; }
        };

        struct FinalExpOp{
            enum : std::size_t{ host_parallel_from = 2 };

            const Fp12* in;
            Fp12* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = detail::final_exp(in[i]); }
        };

        template <Engine E>
        LineStorage<E> prepare_storage(const Vec<G2, E>& qs){
            Buffer<detail::Line, E> lines(qs.size() * detail::kLineCount);
            detail::for_each<E>(qs.size(), PrepareLinesOp{data(qs), lines.data()});
            return {qs, std::move(lines)};
        }

        struct CopyOp{
            const Fp12* in;
            Fp12* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = in[i]; }
        };

        template <Engine E>
        Buffer<Fp12, E> miller_products(
            const detail::G1Affine* ps, const detail::G2Affine* qs, const detail::Line* lines, const PairShape& shape
        ){
            const auto cap = std::same_as<E, Cpu> ? shape.length : detail::kMaxPairsPerItem;
            const auto threads = detail::resident_threads<E, MillerItemOp>();
            const detail::Chunks items{
                shape.segments, shape.length, detail::chunk_size(shape.segments * shape.length, threads, cap)
            };
            Buffer<Fp12, E> products(items.count());
            const MillerItemOp op{
                items, detail::layout(shape, shape.p), detail::layout(shape, shape.q), ps, qs, lines, products.data()
            };
            detail::for_each<E>(items.count(), op);
            const detail::Chunks level{shape.segments, items.per_segment(), detail::kReduceFanIn};
            return detail::reduce_segments<E, Fp12>(std::move(products), level, Fp12Product{});
        }

        template <Engine E>
        Vec<Gt, E> final_exps(const Buffer<Fp12, E>& products){
            return detail::generate<Gt, E>(products.size(), [&](auto* out){ return FinalExpOp{products.data(), out}; });
        }

        template <Engine E>
        Vec<Gt, E> finish_pairing(const Buffer<Fp12, E>& products, const PairShape& shape){
            if constexpr (std::same_as<E, Gpu>){
                if (detail::place(shape) == detail::Placement::host_final_exp){
                    return final_exps<Cpu>(detail::to_host(products)).template to<Gpu>();
                }
            }
            return final_exps<E>(products);
        }

        template <Engine E>
        Vec<Gt, E> pair_prepared(const Vec<G1, E>& ps, const LineStorage<E>& qs, const PairShape& shape){
            return finish_pairing<E>(miller_products<E>(data(ps), data(qs.sources), qs.lines.data(), shape), shape);
        }

        Vec<Gt, Gpu> pair_on_host(const Vec<G1, Gpu>& ps, const LineStorage<Cpu>& qs, const PairShape& shape){
            return pair_prepared(ps.to<Cpu>(), qs, shape).to<Gpu>();
        }

        template <Engine E>
        void require_shapes(const Vec<G1, E>& ps, const std::size_t q_count, const PairShape& shape){
            if (shape.length == 0) throw ShapeError("pair_segments needs a positive segment length");
            (void)detail::checked_product(shape.segments, shape.length, "pair_segments");
            detail::layout(shape, shape.p).require_size(ps.size(), "pair_segments' G1 side");
            detail::layout(shape, shape.q).require_size(q_count, "pair_segments' G2 side");
        }
    }

    template <Engine E>
    G2Lines<E>::G2Lines(std::shared_ptr<const LineStorage<E>> storage) : storage_(std::move(storage)){}

    template <Engine E>
    std::size_t G2Lines<E>::size() const{
        return storage_->sources.size();
    }

    template <Engine E>
    G2Lines<E> prepare(const Vec<G2, E>& qs){
        auto storage = prepare_storage(qs);
        detail::finish<E>();
        return Access::wrap<G2Lines<E>>(std::make_shared<const LineStorage<E>>(std::move(storage)));
    }

    template <Engine E>
    Vec<Gt, E> pair_segments(const Vec<G1, E>& ps, const G2Lines<E>& qs, const PairShape& shape){
        require_shapes(ps, qs.size(), shape);
        const auto& lines = Access::storage(qs);
        if constexpr (std::same_as<E, Gpu>){
            if (detail::place(shape) == detail::Placement::host){
                return pair_on_host(ps, {lines.sources.template to<Cpu>(), detail::to_host(lines.lines)}, shape);
            }
        }
        return pair_prepared(ps, lines, shape);
    }

    template <Engine E>
    Vec<Gt, E> pair_segments(const Vec<G1, E>& ps, const Vec<G2, E>& qs, const PairShape& shape){
        require_shapes(ps, qs.size(), shape);
        if constexpr (std::same_as<E, Gpu>){
            if (detail::place(shape) == detail::Placement::host){
                return pair_on_host(ps, prepare_storage(qs.template to<Cpu>()), shape);
            }
        }
        constexpr auto q_bytes = detail::kLineCount * sizeof(detail::Line);
        const auto chunk = detail::segments_per_line_chunk(shape, q_bytes, detail::kLineBudgetBytes);
        return detail::pair_in_chunks(ps, qs, shape, chunk);
    }

    template <Engine E>
    Vec<Gt, E> detail::pair_in_chunks(
        const Vec<G1, E>& ps, const Vec<G2, E>& qs, const PairShape& shape, const std::size_t chunk_segments
    ){
        if (shape.q == Spread::shared || chunk_segments >= shape.segments){
            return pair_prepared(ps, prepare_storage(qs), shape);
        }
        Buffer<Fp12, E> products(shape.segments);
        const auto p_layout = layout(shape, shape.p);
        for (std::size_t first = 0; first < shape.segments; first += chunk_segments){
            const PairShape part{std::min(chunk_segments, shape.segments - first), shape.length, shape.p, shape.q};
            const auto q_count = part.segments * part.length;
            const auto* q = data(qs) + first * shape.length;
            Buffer<Line, E> lines(q_count * kLineCount);
            for_each<E>(q_count, PrepareLinesOp{q, lines.data()});
            const auto part_products = miller_products<E>(data(ps) + p_layout.offset(first), q, lines.data(), part);
            for_each<E>(part.segments, CopyOp{part_products.data(), products.data() + first});
        }
        return finish_pairing<E>(products, shape);
    }

    template class G2Lines<Cpu>;
    template class G2Lines<Gpu>;
    template G2Lines<Cpu> prepare(const Vec<G2, Cpu>&);
    template G2Lines<Gpu> prepare(const Vec<G2, Gpu>&);
    template Vec<Gt, Cpu> pair_segments(const Vec<G1, Cpu>&, const G2Lines<Cpu>&, const PairShape&);
    template Vec<Gt, Gpu> pair_segments(const Vec<G1, Gpu>&, const G2Lines<Gpu>&, const PairShape&);
    template Vec<Gt, Cpu> pair_segments(const Vec<G1, Cpu>&, const Vec<G2, Cpu>&, const PairShape&);
    template Vec<Gt, Gpu> pair_segments(const Vec<G1, Gpu>&, const Vec<G2, Gpu>&, const PairShape&);
    template Vec<Gt, Cpu> detail::pair_in_chunks(
        const Vec<G1, Cpu>&, const Vec<G2, Cpu>&, const PairShape&, std::size_t);
    template Vec<Gt, Gpu> detail::pair_in_chunks(
        const Vec<G1, Gpu>&, const Vec<G2, Gpu>&, const PairShape&, std::size_t);
}
