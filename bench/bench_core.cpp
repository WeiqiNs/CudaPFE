#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <blst.h>
#include <cufe/cufe.hpp>
#include "dlog/bsgs.hpp"
#include "pairing/plan.hpp"

using namespace cufe;

namespace{
    constexpr int kRuns = 5;
    constexpr double kSingleRunMs = 10000;

    struct Sizes{
        std::vector<std::size_t> pairs;
        std::vector<std::size_t> lengths;
        std::vector<std::size_t> batch_segments;
        std::vector<std::size_t> batch_lengths;
        std::vector<std::size_t> fixed_base;
        std::vector<std::size_t> matrices;
        std::size_t cpu_matrix_limit;
        std::vector<int> dlog_bits;
        std::vector<std::size_t> dlog_batches;
        std::uint64_t cpu_dlog_steps;
        std::size_t cpu_pair_limit;
        std::size_t blst_sample;
    };

    struct Measurement{
        double ms;
        int runs;
    };

    struct Inputs{
        Vec<G1, Gpu> ps;
        Vec<G2, Gpu> qs;
        std::vector<blst_p1_affine> blst_ps;
        std::vector<blst_p2_affine> blst_qs;
    };

    const Sizes kFull{
        {1000, 10000, 100000}, {10, 100, 1000, 10000}, {100, 1000}, {24, 208}, {1000, 10000, 100000, 1000000},
        {100, 1000, 5000}, 1000, {20, 32}, {1, 1000}, std::uint64_t{1} << 24, 25000, 1000,
    };

    const Sizes kQuick{
        {100, 1000}, {10, 100}, {10}, {24}, {1000, 10000}, {16, 64}, 64, {16}, {1, 100}, std::uint64_t{1} << 24, 2000,
        64,
    };

    unsigned cpu_threads(){
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

    std::string cell(const Measurement& m, const double scale = 1){
        const auto value = std::format("{:.2f}", m.ms * scale);
        return m.runs == 1 ? value + " (1 run)" : value;
    }

    std::string rate(const Measurement& m, const std::size_t items){
        return std::format("{:.0f}", static_cast<double>(items) / (m.ms / 1000));
    }

    template <class F>
    void parallel_chunks(const std::size_t count, const F& f){
        const auto threads = cpu_threads();
        const auto per_thread = (count + threads - 1) / threads;
        std::vector<std::thread> workers;
        for (std::size_t begin = 0; begin < count; begin += per_thread){
            workers.emplace_back(f, begin, std::min(count, begin + per_thread));
        }
        for (auto& worker : workers) worker.join();
    }

    void require_match(const Bytes& ours, const Bytes& theirs, const std::string_view what){
        if (ours != theirs) throw std::runtime_error(std::format("{} disagrees with blst", what));
    }

    blst_p1_affine to_blst(const G1& p){
        blst_p1_affine out;
        if (blst_p1_deserialize(&out, p.to_bytes(Encoding::uncompressed).data()) != BLST_SUCCESS){
            throw std::runtime_error("blst rejected a G1 point");
        }
        return out;
    }

    blst_p2_affine to_blst(const G2& q){
        blst_p2_affine out;
        if (blst_p2_deserialize(&out, q.to_bytes(Encoding::uncompressed).data()) != BLST_SUCCESS){
            throw std::runtime_error("blst rejected a G2 point");
        }
        return out;
    }

    blst_scalar to_blst(const Zp& k){
        blst_scalar out;
        blst_scalar_from_bendian(&out, k.to_bytes().data());
        return out;
    }

    Bytes gt_bytes(const blst_fp12& f){
        Bytes out(Gt::byte_size);
        blst_bendian_from_fp12(out.data(), &f);
        return out;
    }

    blst_fp12 blst_pair(const blst_p1_affine& p, const blst_p2_affine& q){
        blst_fp12 miller;
        blst_miller_loop(&miller, &q, &p);
        blst_fp12 out;
        blst_final_exp(&out, &miller);
        return out;
    }

    blst_fp12 blst_miller_n(const Inputs& inputs, const std::size_t first, const std::size_t count){
        std::vector<const blst_p1_affine*> ps;
        std::vector<const blst_p2_affine*> qs;
        for (std::size_t i = first; i < first + count; ++i){
            ps.push_back(&inputs.blst_ps[i]);
            qs.push_back(&inputs.blst_qs[i]);
        }
        blst_fp12 out;
        blst_miller_loop_n(&out, qs.data(), ps.data(), count);
        return out;
    }

    blst_fp12 blst_multi_pair(const Inputs& inputs, const std::size_t first, const std::size_t count){
        const auto miller = blst_miller_n(inputs, first, count);
        blst_fp12 out;
        blst_final_exp(&out, &miller);
        return out;
    }

    blst_fp12 blst_parallel_multi_pair(const Inputs& inputs, const std::size_t count){
        std::vector<blst_fp12> partials(cpu_threads(), *blst_fp12_one());
        const auto per_thread = (count + cpu_threads() - 1) / cpu_threads();
        parallel_chunks(count, [&](const std::size_t begin, const std::size_t end){
            partials[begin / per_thread] = blst_miller_n(inputs, begin, end - begin);
        });
        auto product = *blst_fp12_one();
        for (const auto& partial : partials) blst_fp12_mul(&product, &product, &partial);
        blst_fp12 out;
        blst_final_exp(&out, &product);
        return out;
    }

    Inputs make_inputs(const std::size_t count, const std::size_t blst_count){
        Inputs inputs;
        inputs.ps = mul_generator<G1>(Vec<Zp, Gpu>::upload(random_vector(count)));
        inputs.qs = mul_generator<G2>(Vec<Zp, Gpu>::upload(random_vector(count)));
        const auto ps = inputs.ps.download();
        const auto qs = inputs.qs.download();
        for (std::size_t i = 0; i < std::min(count, blst_count); ++i){
            inputs.blst_ps.push_back(to_blst(ps[i]));
            inputs.blst_qs.push_back(to_blst(qs[i]));
        }
        return inputs;
    }

    void print_header(const std::string_view mode){
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) throw std::runtime_error("cudaGetDeviceProperties");
        std::cout << std::format(
            "# LibCuFE benchmark ({})\n\n"
            "- GPU: {}, {} SMs\n- CPU threads: {}\n- Build type: {}\n"
            "- CUFE_HOST_MILLER_BELOW = {}, CUFE_HOST_FINAL_EXP_BELOW = {}\n"
            "- Median of {} runs after one warm-up; a case whose warm-up exceeds {:.0f} s reports that single run.\n",
            mode, properties.name, properties.multiProcessorCount, cpu_threads(), CUFE_BENCH_BUILD_TYPE,
            detail::kHostMillerBelow, detail::kHostFinalExpBelow, kRuns, kSingleRunMs / 1000
        );
    }

    void pairing_throughput(const Sizes& sizes){
        const auto largest = sizes.pairs.back();
        const auto sample = std::min(largest, sizes.blst_sample);
        const auto parallel_sample = std::min(largest, sizes.blst_sample * cpu_threads());
        const auto inputs = make_inputs(largest, parallel_sample);
        const auto host_ps = inputs.ps.download();
        const auto host_qs = inputs.qs.download();
        std::vector<blst_fp12> sink(parallel_sample);

        const auto single = measure([&]{
            for (std::size_t i = 0; i < sample; ++i) sink[i] = blst_pair(inputs.blst_ps[i], inputs.blst_qs[i]);
        });
        const auto all = measure([&]{
            parallel_chunks(parallel_sample, [&](const std::size_t begin, const std::size_t end){
                for (auto i = begin; i < end; ++i) sink[i] = blst_pair(inputs.blst_ps[i], inputs.blst_qs[i]);
            });
        });

        std::cout << std::format(
            "\n## 1. Pairing throughput (S segments of length 1, pairs/s)\n\n"
            "blst columns are measured on {} (1 core) and {} (all cores) pairs.\n\n"
            "| S | Gpu | blst 1 core | blst all cores |\n| ---: | ---: | ---: | ---: |\n", sample, parallel_sample
        );
        for (const auto s : sizes.pairs){
            const auto ps = Vec<G1, Gpu>::upload(std::span(host_ps).first(s));
            const auto qs = Vec<G2, Gpu>::upload(std::span(host_qs).first(s));
            Vec<Gt, Gpu> out;
            const auto gpu = measure([&]{ out = pair_segments(ps, qs, PairShape{s, 1}); });
            require_match(out.at(0).to_bytes(), gt_bytes(sink[0]), "pair_segments<Gpu>");
            std::cout << std::format("| {} | {} | {} | {} |\n", s, rate(gpu, s), rate(single, sample),
                rate(all, parallel_sample));
        }
    }

    void multi_pairing_latency(const Sizes& sizes){
        const auto largest = sizes.lengths.back();
        const auto inputs = make_inputs(largest, largest);
        const auto host_ps = inputs.ps.download();
        const auto host_qs = inputs.qs.download();
        std::cout << "\n## 2. Multi-pairing latency (1 segment of n pairs, ms)\n\n"
            "| n | Gpu | Cpu engine | blst 1 core | blst all cores |\n| ---: | ---: | ---: | ---: | ---: |\n";
        for (const auto n : sizes.lengths){
            const auto gpu_ps = Vec<G1, Gpu>::upload(std::span(host_ps).first(n));
            const auto gpu_qs = Vec<G2, Gpu>::upload(std::span(host_qs).first(n));
            const auto cpu_ps = Vec<G1, Cpu>::upload(std::span(host_ps).first(n));
            const auto cpu_qs = Vec<G2, Cpu>::upload(std::span(host_qs).first(n));
            Vec<Gt, Gpu> gpu_out;
            Vec<Gt, Cpu> cpu_out;
            blst_fp12 single_out;
            blst_fp12 all_out;
            const auto gpu = measure([&]{ gpu_out = pair_segments(gpu_ps, gpu_qs, PairShape{1, n}); });
            const auto cpu = measure([&]{ cpu_out = pair_segments(cpu_ps, cpu_qs, PairShape{1, n}); });
            const auto single = measure([&]{ single_out = blst_multi_pair(inputs, 0, n); });
            const auto all = measure([&]{ all_out = blst_parallel_multi_pair(inputs, n); });
            require_match(gpu_out.at(0).to_bytes(), gt_bytes(single_out), "pair_segments<Gpu>");
            require_match(cpu_out.at(0).to_bytes(), gt_bytes(all_out), "pair_segments<Cpu>");
            std::cout << std::format("| {} | {} | {} | {} | {} |\n", n, cell(gpu), cell(cpu), cell(single), cell(all));
        }
    }

    void batch_shape(const Sizes& sizes){
        std::cout << std::format(
            "\n## 3. Batch shape (S segments of m pairs, zip layout, ms)\n\n"
            "blst columns time about {} pairs per core and scale to S; the Cpu engine runs up to {} pairs.\n\n"
            "| S | m | Gpu | Cpu engine | blst 1 core | blst all cores |\n"
            "| ---: | ---: | ---: | ---: | ---: | ---: |\n",
            sizes.blst_sample, sizes.cpu_pair_limit
        );
        for (const auto s : sizes.batch_segments){
            for (const auto m : sizes.batch_lengths){
                const auto per_core = std::min(s, sizes.blst_sample / m + 1);
                const auto parallel_segments = std::min(s, per_core * cpu_threads());
                const auto inputs = make_inputs(s * m, parallel_segments * m);
                const PairShape shape{s, m};
                Vec<Gt, Gpu> out;
                const auto gpu = measure([&]{ out = pair_segments(inputs.ps, inputs.qs, shape); });

                std::string cpu_cell = "–";
                if (s * m <= sizes.cpu_pair_limit){
                    const auto ps = Vec<G1, Cpu>::upload(inputs.ps.download());
                    const auto qs = Vec<G2, Cpu>::upload(inputs.qs.download());
                    cpu_cell = cell(measure([&]{ (void)pair_segments(ps, qs, shape); }));
                }

                std::vector<blst_fp12> sink(parallel_segments);
                const auto single = measure([&]{
                    for (std::size_t g = 0; g < per_core; ++g) sink[g] = blst_multi_pair(inputs, g * m, m);
                });
                const auto all = measure([&]{
                    parallel_chunks(parallel_segments, [&](const std::size_t begin, const std::size_t end){
                        for (auto g = begin; g < end; ++g) sink[g] = blst_multi_pair(inputs, g * m, m);
                    });
                });
                require_match(out.at(0).to_bytes(), gt_bytes(sink[0]), "pair_segments<Gpu>");
                std::cout << std::format("| {} | {} | {} | {} | {} | {} |\n", s, m, cell(gpu), cpu_cell,
                    cell(single, static_cast<double>(s) / static_cast<double>(per_core)),
                    cell(all, static_cast<double>(s) / static_cast<double>(parallel_segments)));
            }
        }
    }

    template <class G>
    void fixed_base_row(const Sizes& sizes, const std::string_view name){
        constexpr bool is_g1 = std::same_as<G, G1>;
        const auto largest = sizes.fixed_base.back();
        const auto scalars = random_vector(largest);
        const auto sample = std::min(largest, sizes.blst_sample);
        const auto parallel_sample = std::min(largest, sizes.blst_sample * cpu_threads());
        std::vector<blst_scalar> blst_scalars;
        for (std::size_t i = 0; i < parallel_sample; ++i) blst_scalars.push_back(to_blst(scalars[i]));
        std::vector<Bytes> sink(parallel_sample);
        const auto multiply = [&](const std::size_t i){
            Bytes out(G::compressed_size);
            if constexpr (is_g1){
                blst_p1 p;
                blst_p1_mult(&p, blst_p1_generator(), blst_scalars[i].b, 255);
                blst_p1_affine affine;
                blst_p1_to_affine(&affine, &p);
                blst_p1_affine_compress(out.data(), &affine);
            } else {
                blst_p2 p;
                blst_p2_mult(&p, blst_p2_generator(), blst_scalars[i].b, 255);
                blst_p2_affine affine;
                blst_p2_to_affine(&affine, &p);
                blst_p2_affine_compress(out.data(), &affine);
            }
            sink[i] = std::move(out);
        };
        const auto single = measure([&]{
            for (std::size_t i = 0; i < sample; ++i) multiply(i);
        });
        const auto all = measure([&]{
            parallel_chunks(parallel_sample, [&](const std::size_t begin, const std::size_t end){
                for (auto i = begin; i < end; ++i) multiply(i);
            });
        });
        for (const auto n : sizes.fixed_base){
            const auto gpu_scalars = Vec<Zp, Gpu>::upload(std::span(scalars).first(n));
            Vec<G, Gpu> out;
            const auto gpu = measure([&]{ out = mul_generator<G>(gpu_scalars); });
            require_match(out.at(0).to_bytes(), sink[0], "mul_generator<Gpu>");
            std::cout << std::format("| {} | {} | {} | {} | {} |\n", name, n, cell(gpu),
                cell(single, static_cast<double>(n) / static_cast<double>(sample)),
                cell(all, static_cast<double>(n) / static_cast<double>(parallel_sample)));
        }
    }

    void fixed_base(const Sizes& sizes){
        std::cout << std::format(
            "\n## 4. Fixed-base multiplication (N scalars resident, ms)\n\n"
            "blst runs variable-base blst_p?_mult on the generator plus affine conversion, timed on at most {} (1 "
            "core) and {} (all cores) scalars and scaled to N.\n\n"
            "| Group | N | Gpu | blst 1 core (variable-base) | blst all cores (variable-base) |\n"
            "| --- | ---: | ---: | ---: | ---: |\n",
            sizes.blst_sample, sizes.blst_sample * cpu_threads()
        );
        fixed_base_row<G1>(sizes, "G1");
        fixed_base_row<G2>(sizes, "G2");
    }

    template <Engine E>
    Measurement time_inverse(const std::vector<Zp>& entries, const std::size_t m){
        const auto a = Matrix<E>::upload({m, m}, entries);
        return measure([&]{ (void)a.inverse_with_determinant(); });
    }

    void gauss_jordan(const Sizes& sizes){
        std::cout << std::format(
            "\n## 5. Matrix setup (m x m, ms)\n\n"
            "The Cpu engine runs up to m = {}.\n\n"
            "| m | Gpu inverse | Cpu inverse | Gpu product |\n| ---: | ---: | ---: | ---: |\n", sizes.cpu_matrix_limit
        );
        for (const auto m : sizes.matrices){
            const auto entries = random_vector(m * m);
            const auto a = Matrix<Gpu>::upload({m, m}, entries);
            if (m == sizes.matrices.front() && !(a * a.inverse()).is_identity()){
                throw std::runtime_error("Matrix<Gpu>::inverse is not an inverse");
            }
            const auto gpu = time_inverse<Gpu>(entries, m);
            const auto cpu = m <= sizes.cpu_matrix_limit ? cell(time_inverse<Cpu>(entries, m)) : std::string("–");
            const auto product = measure([&]{ (void)(a * a); });
            std::cout << std::format("| {} | {} | {} | {} |\n", m, cell(gpu), cpu, cell(product));
        }
    }

    template <Engine E>
    std::pair<Measurement, Measurement> time_dlog(const Gt& base, const Range& range, const std::vector<Gt>& targets,
        const std::vector<std::optional<std::int64_t>>& expected){
        std::optional<DlogTable<E>> table;
        const auto build = measure([&]{ table.emplace(base, range); });
        const auto uploaded = Vec<Gt, E>::upload(targets);
        std::vector<std::optional<std::int64_t>> found;
        const auto search = measure([&]{ found = table->find(uploaded); });
        if (found != expected) throw std::runtime_error("DlogTable::find returned a wrong exponent");
        return {build, search};
    }

    void discrete_log(const Sizes& sizes){
        std::cout << std::format(
            "\n## 6. Discrete log (DlogTable over [0, 2^bits], ms)\n\n"
            "The Cpu engine runs when the batch takes at most {} giant steps.\n\n"
            "| bits | batch | Gpu build | Gpu find | Cpu build | Cpu find |\n"
            "| ---: | ---: | ---: | ---: | ---: | ---: |\n",
            sizes.cpu_dlog_steps
        );
        const auto base = Gt::random();
        for (const auto bits : sizes.dlog_bits){
            const Range range{0, std::int64_t{1} << bits};
            const auto giant = detail::plan_steps(range).giant;
            for (const auto batch : sizes.dlog_batches){
                std::vector<Gt> targets;
                std::vector<std::optional<std::int64_t>> expected;
                for (std::size_t i = 0; i < batch; ++i){
                    const auto k = static_cast<std::int64_t>(((i + 1) * 0x9e3779b97f4a7c15ull) >> (64 - bits));
                    targets.push_back(base.pow(k));
                    expected.push_back(k);
                }
                const auto [gpu_build, gpu_find] = time_dlog<Gpu>(base, range, targets, expected);
                std::string cpu_cells = "– | –";
                if (giant * batch <= sizes.cpu_dlog_steps){
                    const auto [cpu_build, cpu_find] = time_dlog<Cpu>(base, range, targets, expected);
                    cpu_cells = cell(cpu_build) + " | " + cell(cpu_find);
                }
                std::cout << std::format("| {} | {} | {} | {} | {} |\n", bits, batch, cell(gpu_build), cell(gpu_find),
                    cpu_cells);
            }
        }
    }

    void placement_sweep(){
        std::cout << "\n## Placement sweep (S segments of n pairs, ms)\n\n"
            "| S | n | pairs | Gpu | Cpu engine | faster |\n| ---: | ---: | ---: | ---: | ---: | --- |\n";
        constexpr std::size_t kLargest = 256;
        constexpr std::size_t kPairLimit = 1024;
        const auto inputs = make_inputs(kPairLimit, 0);
        const auto host_ps = inputs.ps.download();
        const auto host_qs = inputs.qs.download();
        for (std::size_t s = 1; s <= kLargest; s *= 2){
            for (std::size_t n = 1; n <= kLargest && s * n <= kPairLimit; n *= 2){
                const PairShape shape{s, n};
                const auto pairs = s * n;
                const auto gpu_ps = Vec<G1, Gpu>::upload(std::span(host_ps).first(pairs));
                const auto gpu_qs = Vec<G2, Gpu>::upload(std::span(host_qs).first(pairs));
                const auto cpu_ps = Vec<G1, Cpu>::upload(std::span(host_ps).first(pairs));
                const auto cpu_qs = Vec<G2, Cpu>::upload(std::span(host_qs).first(pairs));
                Vec<Gt, Gpu> gpu_out;
                Vec<Gt, Cpu> cpu_out;
                const auto gpu = measure([&]{ gpu_out = pair_segments(gpu_ps, gpu_qs, shape); });
                const auto cpu = measure([&]{ cpu_out = pair_segments(cpu_ps, cpu_qs, shape); });
                if (gpu_out.at(0) != cpu_out.at(0)) throw std::runtime_error("engines disagree in the placement sweep");
                std::cout << std::format("| {} | {} | {} | {} | {} | {} |\n", s, n, pairs, cell(gpu), cell(cpu),
                    gpu.ms < cpu.ms ? "Gpu" : "Cpu");
            }
        }
    }
}

int main(const int argc, char** argv){
    bool quick = false;
    bool sweep = false;
    for (int i = 1; i < argc; ++i){
        const std::string_view argument = argv[i];
        if (argument == "--quick") quick = true;
        else if (argument == "--placement-sweep") sweep = true;
        else {
            std::cerr << "usage: cufe_bench [--quick] [--placement-sweep]\n";
            return 2;
        }
    }
    if (!gpu_available()){
        std::cerr << "cufe_bench needs a CUDA device\n";
        return 1;
    }
    std::cout << std::unitbuf;
    seed(bytes_of("cufe_bench"));
    (void)Gt::generator();

    if (sweep){
        print_header("placement sweep");
        placement_sweep();
        return 0;
    }
    const auto& sizes = quick ? kQuick : kFull;
    print_header(quick ? "quick" : "full");
    pairing_throughput(sizes);
    multi_pairing_latency(sizes);
    batch_shape(sizes);
    fixed_base(sizes);
    gauss_jordan(sizes);
    discrete_log(sizes);
}
