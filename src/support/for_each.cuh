#ifndef CUDAPFE_SUPPORT_FOR_EACH_CUH
#define CUDAPFE_SUPPORT_FOR_EACH_CUH

#include <concepts>
#include <cstddef>
#include <string>
#include <cudapfe/engine.hpp>
#include "support/runtime.hpp"

namespace cudapfe::detail{
    inline constexpr std::size_t kMaxBlocks = (std::size_t{1} << 31) - 1;

    template <class Op>
    consteval int threads_per_block(){
        if constexpr (requires{ Op::threads; }) return Op::threads;
        else return 128;
    }

    template <int Threads, class Op>
    __global__ __launch_bounds__(Threads) void run_each(const std::size_t count, const Op op){
        const auto index = static_cast<std::size_t>(blockIdx.x) * Threads + threadIdx.x;
        if (index < count) op(index);
    }

    template <class Op>
    [[nodiscard]] std::size_t resident_threads(){
        constexpr auto threads = threads_per_block<Op>();
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, run_each<threads, Op>, threads, 0),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
        return static_cast<std::size_t>(blocks) * threads * static_cast<std::size_t>(GpuRuntime::require().sm_count());
    }

    template <Engine E, class Op>
    void for_each(const std::size_t count, const Op& op){
        if constexpr (std::same_as<E, Cpu>){
            for (std::size_t index = 0; index < count; ++index) op(index);
        } else {
            if (count == 0) return;
            constexpr auto threads = threads_per_block<Op>();
            const auto blocks = (count + threads - 1) / threads;
            if (blocks > kMaxBlocks){
                throw DeviceError("for_each over " + std::to_string(count) + " indices exceeds the grid");
            }
            run_each<threads><<<static_cast<unsigned>(blocks), threads, 0, GpuRuntime::require().stream()>>>(count, op);
            check(cudaGetLastError(), "for_each launch");
        }
    }
}

#endif
