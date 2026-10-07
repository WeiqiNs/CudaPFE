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
#include <cudapfe/cudapfe.hpp>
#include "dlog/bsgs.hpp"
#include "pairing/plan.hpp"
#include "timing.hpp"

using namespace cudapfe;
using namespace cudapfe::bench;

namespace{
    struct Sizes{
        std::vector<std::size_t> lengths;
        std::vector<PairShape> batch_shapes;
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

    struct Samples{
        std::size_t one_core;
        std::size_t all_cores;
    };

    struct BlstTimes{
        Measurement one_core;
        Measurement all_cores;
    };

    const Sizes kFull{
        .lengths = {10, 100, 1000, 10000},
        .batch_shapes = {{1000, 1}, {10000, 1}, {100000, 1}, {100, 24}, {100, 208}, {1000, 24}, {1000, 208}},
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
        .lengths = {10, 100},
        .batch_shapes = {{100, 1}, {1000, 1}, {10, 24}},
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

    template <class Out, class P>
    Out deserialized(const P& p, BLST_ERROR (*deserialize)(Out*, const byte*)){
        Out out;
        if (deserialize(&out, p.to_bytes(Encoding::uncompressed).data()) != BLST_SUCCESS){
            throw std::runtime_error("blst rejected a point");
        }
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
            inputs.blst_ps.push_back(deserialized(inputs.ps[i], blst_p1_deserialize));
            inputs.blst_qs.push_back(deserialized(inputs.qs[i], blst_p2_deserialize));
        }
        return inputs;
    }

    template <Engine E>
    Timed<Gt> time_pairing(const Inputs& inputs, const PairShape& shape){
        const auto pairs = shape.segments * shape.length;
        const auto ps = Vec<G1, E>::upload(std::span(inputs.ps).first(pairs));
        const auto qs = Vec<G2, E>::upload(std::span(inputs.qs).first(pairs));
        const auto [time, out] = timed([&]{ return pair_segments(ps, qs, shape); });
        return {time, out.at(0)};
    }

    void print_header(const std::string_view mode){
        std::cout << std::format("# CudaPFE benchmark ({})\n\n", mode) << machine_summary()
            << std::format("- CUDAPFE_HOST_MILLER_BELOW = {}, CUDAPFE_HOST_FINAL_EXP_BELOW = {}\n", detail::kHostMillerBelow,
                detail::kHostFinalExpBelow)
            << timing_summary();
    }

    void multi_pairing_latency(const Sizes& sizes){
        const auto largest = sizes.lengths.back();
        const auto inputs = make_inputs(largest, largest);
        std::cout << "\n## 1. Multi-pairing latency (1 segment of n pairs, ms)\n\n"
            "| n | Gpu | Cpu engine | blst 1 core | blst all cores |\n| ---: | ---: | ---: | ---: | ---: |\n";
        for (const auto n : sizes.lengths){
            const PairShape shape{1, n};
            const auto gpu = time_pairing<Gpu>(inputs, shape);
            const auto cpu = time_pairing<Cpu>(inputs, shape);
            const auto single = timed([&]{ return blst_multi_pair(inputs, 0, n); });
            const auto all = timed([&]{ return blst_parallel_multi_pair(inputs, n); });
            require_match(gpu.result.to_bytes(), gt_bytes(single.result), "pair_segments<Gpu>");
            require_match(cpu.result.to_bytes(), gt_bytes(all.result), "pair_segments<Cpu>");
            std::cout << std::format("| {} | {} | {} | {} | {} |\n", n, cell(gpu.time), cell(cpu.time),
                cell(single.time), cell(all.time));
        }
    }

    void batch_shape(const Sizes& sizes){
        std::cout << std::format(
            "\n## 2. Batch shape (S segments of m pairs, zip layout, ms)\n\n"
            "blst columns time about {} pairs per core and scale to S; the Cpu engine runs up to {} pairs.\n\n"
            "| S | m | Gpu | Cpu engine | blst 1 core | blst all cores |\n"
            "| ---: | ---: | ---: | ---: | ---: | ---: |\n",
            sizes.blst_sample, sizes.cpu_pair_limit
        );
        for (const auto& shape : sizes.batch_shapes){
            const auto s = shape.segments;
            const auto m = shape.length;
            const auto samples = blst_samples(s, sizes.blst_sample / m + 1);
            const auto inputs = make_inputs(s * m, samples.all_cores * m);
            const auto gpu = time_pairing<Gpu>(inputs, shape);
            const auto cpu_cell = s * m <= sizes.cpu_pair_limit ? cell(time_pairing<Cpu>(inputs, shape).time)
                : std::string("–");

            std::vector<blst_fp12> sink(samples.all_cores);
            const auto blst = time_blst(samples, [&](const std::size_t g){
                sink[g] = blst_multi_pair(inputs, g * m, m);
            });
            require_match(gpu.result.to_bytes(), gt_bytes(sink[0]), "pair_segments<Gpu>");
            std::cout << std::format("| {} | {} | {} | {} | {} | {} |\n", s, m, cell(gpu.time), cpu_cell,
                scaled_cell(blst.one_core, s, samples.one_core), scaled_cell(blst.all_cores, s, samples.all_cores));
        }
    }

    template <class G>
    void fixed_base_row(const Sizes& sizes, const std::string_view name){
        constexpr bool is_g1 = std::same_as<G, G1>;
        const auto largest = sizes.fixed_base.back();
        const auto scalars = random_vector(largest);
        const auto samples = blst_samples(largest, sizes.blst_sample);
        std::vector<blst_scalar> blst_scalars(samples.all_cores);
        for (std::size_t i = 0; i < samples.all_cores; ++i){
            blst_scalar_from_bendian(&blst_scalars[i], scalars[i].to_bytes().data());
        }
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
            const auto gpu = timed([&]{ return mul_generator<G>(gpu_scalars); });
            require_match(gpu.result.at(0).to_bytes(), sink[0], "mul_generator<Gpu>");
            std::cout << std::format("| {} | {} | {} | {} | {} |\n", name, n, cell(gpu.time),
                scaled_cell(blst.one_core, n, samples.one_core), scaled_cell(blst.all_cores, n, samples.all_cores));
        }
    }

    void fixed_base(const Sizes& sizes){
        std::cout << std::format(
            "\n## 3. Fixed-base multiplication (N scalars resident, ms)\n\n"
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
            "\n## 4. Matrix setup (m x m, ms)\n\n"
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
        const auto table = timed([&]{ return DlogTable<E>(base, range); });
        const auto uploaded = Vec<Gt, E>::upload(targets);
        const auto search = timed([&]{ return table.result.find(uploaded); });
        if (search.result != expected) throw std::runtime_error("DlogTable::find returned a wrong exponent");
        return {table.time, search.time};
    }

    void discrete_log(const Sizes& sizes){
        std::cout << std::format(
            "\n## 5. Discrete log (DlogTable over [0, 2^bits], ms)\n\n"
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
                if (gpu.result != cpu.result) throw std::runtime_error("engines disagree in the placement sweep");
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
            std::cerr << "usage: cudapfe_bench [--quick] [--placement-sweep]\n";
            return 2;
        }
    }
    if (!gpu_available()){
        std::cerr << "cudapfe_bench needs a CUDA device\n";
        return 1;
    }
    std::cout << std::unitbuf;
    seed(bytes_of("cudapfe_bench"));
    (void)Gt::generator();

    if (sweep){
        print_header("placement sweep");
        placement_sweep();
        return 0;
    }
    const auto& sizes = quick ? kQuick : kFull;
    print_header(quick ? "quick" : "full");
    multi_pairing_latency(sizes);
    batch_shape(sizes);
    fixed_base(sizes);
    gauss_jordan(sizes);
    discrete_log(sizes);
}
