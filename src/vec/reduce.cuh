#ifndef CUDAPFE_VEC_REDUCE_CUH
#define CUDAPFE_VEC_REDUCE_CUH

#include <cstddef>
#include <utility>
#include <cudapfe/engine.hpp>
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/reduction.hpp"
#include "vec/storage.hpp"

namespace cudapfe::detail{
    template <class T, class Combine>
    struct ReduceOp{
        Chunks level;
        const T* in;
        T* out;
        Combine combine;

        CUDAPFE_HD void operator()(const std::size_t i) const{
            const auto group = level.at(i);
            const auto* values = in + group.segment * level.length + group.first;
            auto total = values[0];
#pragma unroll 1
            for (std::size_t k = 1; k < group.count; ++k) total = combine(total, values[k]);
            out[i] = total;
        }
    };

    template <Engine E, class T, class Combine>
    [[nodiscard]] Buffer<T, E> reduce_segments(Buffer<T, E> values, Chunks level, const Combine& combine){
        while (level.length > 1){
            Buffer<T, E> next(level.count());
            for_each<E>(level.count(), ReduceOp<T, Combine>{level, values.data(), next.data(), combine});
            values = std::move(next);
            level.length = level.per_segment();
        }
        return values;
    }
}

#endif
