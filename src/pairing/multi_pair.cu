#include <concepts>
#include <cstddef>
#include <memory>
#include <utility>
#include <cufe/core.hpp>
#include <cufe/pairing.hpp>
#include <cufe/vec.hpp>
#include "curve/curve.hpp"
#include "field/tower.hpp"
#include "pairing/miller.hpp"
#include "pairing/plan.hpp"
#include "support/access.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/reduce.cuh"
#include "vec/storage.hpp"

namespace cufe{
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
            const detail::G2Affine* qs;
            detail::Line* lines;

            CUFE_HD void operator()(const std::size_t i) const{
                detail::prepare_lines(qs[i], lines + i * detail::kLineCount);
            }
        };

        static_assert(detail::kMaxPairsPerItem <= detail::MaskedPairs::capacity);

        struct MillerItemOp{
            detail::ItemPlan plan;
            const detail::G1Affine* ps;
            const detail::G2Affine* qs;
            const detail::Line* lines;
            Fp12* out;

            CUFE_HD void operator()(const std::size_t i) const{
                const auto item = plan.item(i);
                const auto p = detail::segment_offset(plan.shape, plan.shape.p, item.segment) + item.first;
                const auto q = detail::segment_offset(plan.shape, plan.shape.q, item.segment) + item.first;
                const detail::PreparedPairs pairs{ps + p, qs + q, lines + q * detail::kLineCount, item.count};
#ifdef __CUDA_ARCH__
                out[i] = detail::miller(detail::MaskedPairs::gather(pairs));
#else
                out[i] = detail::miller(pairs);
#endif
            }
        };

        struct Fp12Product{
            CUFE_HD Fp12 operator()(const Fp12& x, const Fp12& y) const{ return x * y; }
        };

        struct FinalExpOp{
            const Fp12* in;
            Fp12* out;

            CUFE_HD void operator()(const std::size_t i) const{ out[i] = detail::final_exp(in[i]); }
        };

        template <Engine E>
        LineStorage<E> prepare_storage(const Vec<G2, E>& qs){
            Buffer<detail::Line, E> lines(qs.size() * detail::kLineCount);
            detail::for_each<E>(qs.size(), PrepareLinesOp{data(qs), lines.data()});
            return {qs, std::move(lines)};
        }

        template <Engine E>
        std::size_t pairs_per_item(const PairShape& shape){
            if constexpr (std::same_as<E, Cpu>) return shape.length;
            else return detail::pairs_per_item(shape, detail::resident_threads<MillerItemOp>());
        }

        template <Engine E>
        Buffer<Fp12, E> miller_products(const Vec<G1, E>& ps, const LineStorage<E>& qs, const PairShape& shape){
            const detail::ItemPlan plan{shape, pairs_per_item<E>(shape)};
            Buffer<Fp12, E> items(plan.item_count());
            const MillerItemOp op{plan, data(ps), data(qs.sources), qs.lines.data(), items.data()};
            detail::for_each<E>(plan.item_count(), op);
            return detail::reduce_segments<E, Fp12>(std::move(items), plan.products(), Fp12Product{});
        }

        template <Engine E>
        Vec<Gt, E> final_exps(const Buffer<Fp12, E>& products){
            return detail::generate<Gt, E>(products.size(), [&](auto* out){ return FinalExpOp{products.data(), out}; });
        }

        template <class T>
        Vec<T, Cpu> to_cpu(const Vec<T, Gpu>& values){
            return detail::vec<T, Cpu>(detail::to_host(detail::buffer(values)));
        }

        template <class T>
        Vec<T, Gpu> to_gpu(const Vec<T, Cpu>& values){
            return detail::vec<T, Gpu>(detail::to_engine<Gpu>(detail::buffer(values)));
        }

        template <Engine E>
        Vec<Gt, E> pair_prepared(const Vec<G1, E>& ps, const LineStorage<E>& qs, const PairShape& shape){
            const auto products = miller_products<E>(ps, qs, shape);
            if constexpr (std::same_as<E, Gpu>){
                if (detail::place(shape) == detail::Placement::host_final_exp){
                    return to_gpu(final_exps<Cpu>(detail::to_host(products)));
                }
            }
            return final_exps<E>(products);
        }

        Vec<Gt, Gpu> pair_on_host(const Vec<G1, Gpu>& ps, const LineStorage<Cpu>& qs, const PairShape& shape){
            return to_gpu(pair_prepared(to_cpu(ps), qs, shape));
        }

        template <Engine E>
        void require_shapes(const Vec<G1, E>& ps, const std::size_t q_count, const PairShape& shape){
            detail::require_shape(shape);
            detail::require_side(shape, shape.p, ps.size(), "G1");
            detail::require_side(shape, shape.q, q_count, "G2");
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
                return pair_on_host(ps, {to_cpu(lines.sources), detail::to_host(lines.lines)}, shape);
            }
        }
        return pair_prepared(ps, lines, shape);
    }

    template <Engine E>
    Vec<Gt, E> pair_segments(const Vec<G1, E>& ps, const Vec<G2, E>& qs, const PairShape& shape){
        require_shapes(ps, qs.size(), shape);
        if constexpr (std::same_as<E, Gpu>){
            if (detail::place(shape) == detail::Placement::host){
                return pair_on_host(ps, prepare_storage(to_cpu(qs)), shape);
            }
        }
        return pair_prepared(ps, prepare_storage(qs), shape);
    }

    template class G2Lines<Cpu>;
    template class G2Lines<Gpu>;
    template G2Lines<Cpu> prepare(const Vec<G2, Cpu>&);
    template G2Lines<Gpu> prepare(const Vec<G2, Gpu>&);
    template Vec<Gt, Cpu> pair_segments(const Vec<G1, Cpu>&, const G2Lines<Cpu>&, const PairShape&);
    template Vec<Gt, Gpu> pair_segments(const Vec<G1, Gpu>&, const G2Lines<Gpu>&, const PairShape&);
    template Vec<Gt, Cpu> pair_segments(const Vec<G1, Cpu>&, const Vec<G2, Cpu>&, const PairShape&);
    template Vec<Gt, Gpu> pair_segments(const Vec<G1, Gpu>&, const Vec<G2, Gpu>&, const PairShape&);
}
