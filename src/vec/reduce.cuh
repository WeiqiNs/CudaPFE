#ifndef CUFE_VEC_REDUCE_CUH
#define CUFE_VEC_REDUCE_CUH

#include <cstddef>
#include <utility>
#include <cufe/engine.hpp>
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/reduction.hpp"
#include "vec/storage.hpp"

namespace cufe::detail{
    template <class T, class Combine>
    struct ReduceOp{
        Reduction level;
        const T* in;
        T* out;
        Combine combine;

        CUFE_HD void operator()(const std::size_t i) const{
            const auto group = level.group(i);
            auto total = in[group.first];
#pragma unroll 1
            for (std::size_t k = 1; k < group.count; ++k) total = combine(total, in[group.first + k]);
            out[i] = total;
        }
    };

    template <Engine E, class T, class Combine>
    [[nodiscard]] Buffer<T, E> reduce_segments(Buffer<T, E> values, Reduction level, const Combine& combine){
        while (level.width > 1){
            Buffer<T, E> next(level.group_count());
            for_each<E>(level.group_count(), ReduceOp<T, Combine>{level, values.data(), next.data(), combine});
            values = std::move(next);
            level = level.next();
        }
        return values;
    }
}

#endif
