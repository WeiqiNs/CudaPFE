#ifndef CUFE_BENCH_TIMING_HPP
#define CUFE_BENCH_TIMING_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <stdexcept>
#include <string>
#include <thread>
#include <cuda_runtime.h>

namespace cufe::bench{
    inline constexpr int kRuns = 5;
    inline constexpr double kSingleRunMs = 10000;

    struct Measurement{
        double ms;
        int runs;
    };

    inline unsigned cpu_threads(){
        return std::max(std::thread::hardware_concurrency(), 1u);
    }

    template <class F>
    double elapsed_ms(F& f){
        const auto start = std::chrono::steady_clock::now();
        f();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    template <class F>
    Measurement measure(F&& f){
        const auto warm_up = elapsed_ms(f);
        if (warm_up >= kSingleRunMs) return {warm_up, 1};
        std::array<double, kRuns> times;
        for (auto& time : times) time = elapsed_ms(f);
        std::ranges::sort(times);
        return {times[kRuns / 2], kRuns};
    }

    inline std::string cell(const Measurement& m){
        const auto value = std::format("{:.2f}", m.ms);
        return m.runs == 1 ? value + " (1 run)" : value;
    }

    inline std::string machine_summary(){
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) throw std::runtime_error("cudaGetDeviceProperties");
        return std::format("- GPU: {}, {} SMs\n- CPU threads: {}\n- Build type: {}\n", properties.name,
            properties.multiProcessorCount, cpu_threads(), CUFE_BENCH_BUILD_TYPE);
    }

    inline std::string timing_summary(){
        return std::format(
            "- Median of {} runs after one warm-up; a case whose warm-up exceeds {:.0f} s reports that single run.\n",
            kRuns, kSingleRunMs / 1000
        );
    }
}

#endif
