#include <algorithm>
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
#include <blst.h>
#include <cufe/cufe.hpp>
#include "dlog/bsgs.hpp"
#include "pairing/plan.hpp"
#include "timing.hpp"

using namespace cufe;
using namespace cufe::bench;

namespace{
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

    struct Inputs{
        std::vector<G1> ps;
        std::vector<G2> qs;
        std::vector<blst_p1_affine> blst_ps;
        std::vector<blst_p2_affine> blst_qs;
    };

    struct Timed{
        Measurement time;
        Gt first;
    };

    struct Samples{
        std::size_t one_core;
        std::size_t all_cores;
    };

    struct BlstTimes{
        Measurement one_core;
        Measurement all_cores;
    };

    const Sizes kFull{
        .pairs = {1000, 10000, 100000},
        .lengths = {10, 100, 1000, 10000},
        .batch_segments = {100, 1000},
        .batch_lengths = {24, 208},
        .fixed_base = {1000, 10000, 100000, 1000000},
        .matrices = {100, 1000, 5000},
        .cpu_matrix_limit = 1000,
        .dlog_bits = {20, 32},
        .dlog_batches = {1, 1000},
        .cpu_dlog_steps = std::uint64_t{1} << 24,
        .cpu_pair_limit = 25000,
        .blst_sample = 1000,
    };

    const Sizes kQuick{
        .pairs = {100, 1000},
        .lengths = {10, 100},
        .batch_segments = {10},
        .batch_lengths = {24},
        .fixed_base = {1000, 10000},
        .matrices = {16, 64},
        .cpu_matrix_limit = 64,
        .dlog_bits = {16},
        .dlog_batches = {1, 100},
        .cpu_dlog_steps = std::uint64_t{1} << 24,
        .cpu_pair_limit = 2000,
        .blst_sample = 64,
    };

    std::string scaled_cell(const Measurement& m, const std::size_t items, const std::size_t measured){
        return cell({m.ms * (static_cast<double>(items) / static_cast<double>(measured)), m.runs});
    }

    std::string rate(const Measurement& m, const std::size_t items){
        return std::format("{:.0f}", static_cast<double>(items) / (m.ms / 1000));
    }

    std::size_t chunk_size(const std::size_t count){
        return (count + cpu_threads() - 1) / cpu_threads();
    }

    template <class F>
    void parallel_chunks(const std::size_t count, const F& f){
        const auto chunk = chunk_size(count);
        std::vector<std::thread> workers;
        for (std::size_t begin = 0; begin < count; begin += chunk){
            workers.emplace_back(f, begin, std::min(count, begin + chunk));
        }
        for (auto& worker : workers) worker.join();
    }

    Samples blst_samples(const std::size_t available, const std::size_t per_core){
        return {std::min(available, per_core), std::min(available, per_core * cpu_threads())};
    }

    template <class F>
    BlstTimes time_blst(const Samples& samples, const F& run){
        return {
            measure([&]{
                for (std::size_t i = 0; i < samples.one_core; ++i) run(i);
            }),
            measure([&]{
                parallel_chunks(samples.all_cores, [&](const std::size_t begin, const std::size_t end){
                    for (auto i = begin; i < end; ++i) run(i);
                });
            }),
        };
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
        const auto chunk = chunk_size(count);
        parallel_chunks(count, [&](const std::size_t begin, const std::size_t end){
            partials[begin / chunk] = blst_miller_n(inputs, begin, end - begin);
        });
        auto product = *blst_fp12_one();
        for (const auto& partial : partials) blst_fp12_mul(&product, &product, &partial);
        blst_fp12 out;
        blst_final_exp(&out, &product);
        return out;
    }

    Inputs make_inputs(const std::size_t count, const std::size_t blst_count){
        Inputs inputs;
        inputs.ps = mul_generator<G1>(Vec<Zp, Gpu>::upload(random_vector(count))).download();
        inputs.qs = mul_generator<G2>(Vec<Zp, Gpu>::upload(random_vector(count))).download();
        for (std::size_t i = 0; i < std::min(count, blst_count); ++i){
            inputs.blst_ps.push_back(to_blst(inputs.ps[i]));
            inputs.blst_qs.push_back(to_blst(inputs.qs[i]));
        }
        return inputs;
    }

    template <Engine E>
    Timed time_pairing(const Inputs& inputs, const PairShape& shape){
        const auto pairs = shape.segments * shape.length;
        const auto ps = Vec<G1, E>::upload(std::span(inputs.ps).first(pairs));
        const auto qs = Vec<G2, E>::upload(std::span(inputs.qs).first(pairs));
        Vec<Gt, E> out;
        const auto time = measure([&]{ out = pair_segments(ps, qs, shape); });
        return {time, out.at(0)};
    }

    void print_header(const std::string_view mode){
        std::cout << std::format("# LibCuFE benchmark ({})\n\n", mode) << machine_summary()
            << std::format("- CUFE_HOST_MILLER_BELOW = {}, CUFE_HOST_FINAL_EXP_BELOW = {}\n", detail::kHostMillerBelow,
                detail::kHostFinalExpBelow)
            << timing_summary();
    }

    void pairing_throughput(const Sizes& sizes){
        const auto largest = sizes.pairs.back();
        const auto samples = blst_samples(largest, sizes.blst_sample);
        const auto inputs = make_inputs(largest, samples.all_cores);
        std::vector<blst_fp12> sink(samples.all_cores);
        const auto blst = time_blst(samples, [&](const std::size_t i){
            sink[i] = blst_pair(inputs.blst_ps[i], inputs.blst_qs[i]);
        });

        std::cout << std::format(
            "\n## 1. Pairing throughput (S segments of length 1, pairs/s)\n\n"
            "blst columns are measured on {} (1 core) and {} (all cores) pairs.\n\n"
            "| S | Gpu | blst 1 core | blst all cores |\n| ---: | ---: | ---: | ---: |\n",
            samples.one_core, samples.all_cores
        );
        for (const auto s : sizes.pairs){
            const auto gpu = time_pairing<Gpu>(inputs, {s, 1});
            require_match(gpu.first.to_bytes(), gt_bytes(sink[0]), "pair_segments<Gpu>");
            std::cout << std::format("| {} | {} | {} | {} |\n", s, rate(gpu.time, s),
                rate(blst.one_core, samples.one_core), rate(blst.all_cores, samples.all_cores));
        }
    }

    void multi_pairing_latency(const Sizes& sizes){
        const auto largest = sizes.lengths.back();
        const auto inputs = make_inputs(largest, largest);
        std::cout << "\n## 2. Multi-pairing latency (1 segment of n pairs, ms)\n\n"
            "| n | Gpu | Cpu engine | blst 1 core | blst all cores |\n| ---: | ---: | ---: | ---: | ---: |\n";
        for (const auto n : sizes.lengths){
            const PairShape shape{1, n};
            const auto gpu = time_pairing<Gpu>(inputs, shape);
            const auto cpu = time_pairing<Cpu>(inputs, shape);
            blst_fp12 single_out;
            blst_fp12 all_out;
            const auto single = measure([&]{ single_out = blst_multi_pair(inputs, 0, n); });
            const auto all = measure([&]{ all_out = blst_parallel_multi_pair(inputs, n); });
            require_match(gpu.first.to_bytes(), gt_bytes(single_out), "pair_segments<Gpu>");
            require_match(cpu.first.to_bytes(), gt_bytes(all_out), "pair_segments<Cpu>");
            std::cout << std::format("| {} | {} | {} | {} | {} |\n", n, cell(gpu.time), cell(cpu.time), cell(single),
                cell(all));
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
                const PairShape shape{s, m};
                const auto samples = blst_samples(s, sizes.blst_sample / m + 1);
                const auto inputs = make_inputs(s * m, samples.all_cores * m);
                const auto gpu = time_pairing<Gpu>(inputs, shape);
                const auto cpu_cell = s * m <= sizes.cpu_pair_limit ? cell(time_pairing<Cpu>(inputs, shape).time)
                    : std::string("–");

                std::vector<blst_fp12> sink(samples.all_cores);
                const auto blst = time_blst(samples, [&](const std::size_t g){
                    sink[g] = blst_multi_pair(inputs, g * m, m);
                });
                require_match(gpu.first.to_bytes(), gt_bytes(sink[0]), "pair_segments<Gpu>");
                std::cout << std::format("| {} | {} | {} | {} | {} | {} |\n", s, m, cell(gpu.time), cpu_cell,
                    scaled_cell(blst.one_core, s, samples.one_core), scaled_cell(blst.all_cores, s, samples.all_cores));
            }
        }
    }

    template <class G>
    void fixed_base_row(const Sizes& sizes, const std::string_view name){
        constexpr bool is_g1 = std::same_as<G, G1>;
        const auto largest = sizes.fixed_base.back();
        const auto scalars = random_vector(largest);
        const auto samples = blst_samples(largest, sizes.blst_sample);
        std::vector<blst_scalar> blst_scalars;
        for (std::size_t i = 0; i < samples.all_cores; ++i) blst_scalars.push_back(to_blst(scalars[i]));
        std::vector<Bytes> sink(samples.all_cores);
        const auto blst = time_blst(samples, [&](const std::size_t i){
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
        });
        for (const auto n : sizes.fixed_base){
            const auto gpu_scalars = Vec<Zp, Gpu>::upload(std::span(scalars).first(n));
            Vec<G, Gpu> out;
            const auto gpu = measure([&]{ out = mul_generator<G>(gpu_scalars); });
            require_match(out.at(0).to_bytes(), sink[0], "mul_generator<Gpu>");
            std::cout << std::format("| {} | {} | {} | {} | {} |\n", name, n, cell(gpu),
                scaled_cell(blst.one_core, n, samples.one_core), scaled_cell(blst.all_cores, n, samples.all_cores));
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
        for (std::size_t s = 1; s <= kLargest; s *= 2){
            for (std::size_t n = 1; n <= kLargest && s * n <= kPairLimit; n *= 2){
                const PairShape shape{s, n};
                const auto gpu = time_pairing<Gpu>(inputs, shape);
                const auto cpu = time_pairing<Cpu>(inputs, shape);
                if (gpu.first != cpu.first) throw std::runtime_error("engines disagree in the placement sweep");
                std::cout << std::format("| {} | {} | {} | {} | {} | {} |\n", s, n, s * n, cell(gpu.time),
                    cell(cpu.time), gpu.time.ms < cpu.time.ms ? "Gpu" : "Cpu");
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
