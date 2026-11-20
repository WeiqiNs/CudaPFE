#include <cstddef>
#include <cudapfe/core.hpp>
#include <cudapfe/vec.hpp>
#include "curve/curve.hpp"
#include "curve/generator_table.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/placement.hpp"
#include "vec/points.cuh"
#include "vec/storage.hpp"

namespace cudapfe{
    namespace{
        using detail::data;
        using detail::FieldOf;

        template <class F>
        struct FixedBaseOp{
            enum : std::size_t{ host_parallel_from = 2 };

            const detail::Fr* scalars;
            const detail::Affine<F>* table;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = detail::fixed_base_mul(table, scalars[i]); }
        };

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

    template <class G, Engine E> requires detail::GroupPoint<G>
    Vec<G, E> mul_generator(const Vec<Zp, E>& scalars){
        if constexpr (std::same_as<E, Gpu>){
            if (detail::points_on_host(scalars.size())){
                return mul_generator<G>(scalars.template to<Cpu>()).template to<Gpu>();
            }
        }
        using F = FieldOf<G>;
        const auto* table = engine_generator_table<E, F>();
        return detail::generate_points<G, E>(scalars.size(), [&](auto* out){
            return FixedBaseOp<F>{data(scalars), table, out};
        });
    }

    template Vec<G1, Cpu> mul_generator<G1, Cpu>(const Vec<Zp, Cpu>&);
    template Vec<G1, Gpu> mul_generator<G1, Gpu>(const Vec<Zp, Gpu>&);
    template Vec<G2, Cpu> mul_generator<G2, Cpu>(const Vec<Zp, Cpu>&);
    template Vec<G2, Gpu> mul_generator<G2, Gpu>(const Vec<Zp, Gpu>&);
}
