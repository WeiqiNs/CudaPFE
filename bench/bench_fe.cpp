#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cudapfe/cudapfe.hpp>
#include "schemes.hpp"
#include "timing.hpp"

using namespace cudapfe;
using namespace cudapfe::bench;

namespace{
    constexpr std::int64_t kBound = 10000;
    constexpr std::size_t kCpuMatrixLimit = 1000;
    constexpr std::size_t kCpuPairLimit = 25000;
    constexpr std::size_t kCpuMsmLimit = 50000;
    constexpr std::size_t kQuadraticLengthLimit = 100;
    constexpr std::size_t kJacobianG1Bytes = 3 * 48;

    using IPFE::Results;

    struct Case{
        std::size_t length;
        std::size_t batch;
    };

    struct Sample{
        IPFE::IntMatrix x;
        IPFE::IntMatrix y;
        std::vector<QFE::IntMatrix> f;
        Results values;
    };

    struct Footprint{
        std::size_t pairs;
        std::size_t matrix;
        std::size_t msm_terms;
    };

    const std::vector<Case> kFull{{10, 1}, {100, 1}, {1000, 1}, {10000, 1}, {10, 1000}, {100, 1000}};
    const std::vector<Case> kQuick{{10, 1}, {100, 1}, {10, 16}};

    Sample samples(const Case& c, const int degree){
        std::mt19937_64 engine(c.length * c.batch);
        const auto terms = std::pow(static_cast<double>(c.length), degree - 1);
        const auto largest = static_cast<std::int64_t>(std::pow(static_cast<double>(kBound) / terms, 1.0 / degree));
        std::uniform_int_distribution<std::int64_t> entry(0, largest);
        const auto random_row = [&]{
            IPFE::IntVec v(c.length);
            for (auto& e : v) e = entry(engine);
            return v;
        };
        Sample sample;
        for (std::size_t b = 0; b < c.batch; ++b){
            const auto x = random_row();
            const auto y = random_row();
            std::int64_t value = 0;
            if (degree == 2){
                for (std::size_t i = 0; i < c.length; ++i) value += x[i] * y[i];
            } else {
                QFE::IntMatrix f;
                for (std::size_t i = 0; i < c.length; ++i){
                    f.push_back(random_row());
                    for (std::size_t j = 0; j < c.length; ++j) value += x[i] * f[i][j] * y[j];
                }
                sample.f.push_back(std::move(f));
            }
            sample.x.push_back(x);
            sample.y.push_back(y);
            sample.values.push_back(value);
        }
        return sample;
    }

    struct InnerProduct{
        static constexpr int degree = 2;
        static constexpr std::string_view title = "Inner-product FE";
        static auto key(const auto& msk, const Sample& sample){ return keygen(msk, sample.y); }
        static auto encrypt(const auto& msk, const Sample& sample){ return enc(msk, sample.x); }
    };

    struct Quadratic{
        static constexpr int degree = 3;
        static constexpr std::string_view title = "Quadratic FE";
        static auto key(const auto& keys, const Sample& sample){ return keygen(keys.msk, sample.f); }
        static auto encrypt(const auto& keys, const Sample& sample){ return enc(keys.pk, sample.x, sample.y); }
    };

    template <template <Engine> class Scheme>
    Footprint footprint(std::size_t n);

    template <>
    Footprint footprint<Bjk>(const std::size_t n){ return {2 * n + 6, 2 * n + 4, 0}; }

    template <>
    Footprint footprint<Tao>(const std::size_t n){ return {2 * n + 5, 2 * n + 5, 0}; }

    template <>
    Footprint footprint<Kim>(const std::size_t n){ return {n + 1, n, 0}; }

    template <>
    Footprint footprint<Lin>(const std::size_t n){ return {2 * n + 2, 0, 0}; }

    template <>
    Footprint footprint<Kks>(const std::size_t n){ return {2 * n + 8, 0, 0}; }

    template <>
    Footprint footprint<Opt>(const std::size_t n){ return {n + 4, 0, 0}; }

    template <>
    Footprint footprint<Bcfg>(const std::size_t n){ return {3 * n + 2, 0, n * n}; }

    template <>
    Footprint footprint<Sgp>(const std::size_t n){ return {2 * n + 1, 0, n * n}; }

    template <Engine E>
    constexpr std::string_view engine_name(){
        return std::same_as<E, Gpu> ? "Gpu" : "Cpu";
    }

    template <Engine E>
    std::optional<std::string> skip_reason(const Footprint& footprint, const Case& c){
        const auto pairs = c.batch * footprint.pairs;
        const auto terms = c.batch * footprint.msm_terms;
        if constexpr (std::same_as<E, Cpu>){
            if (footprint.matrix > kCpuMatrixLimit){
                return std::format("setup inverts a {0} x {0} matrix; the Cpu engine runs up to {1} x {1}",
                    footprint.matrix, kCpuMatrixLimit);
            }
            if (pairs > kCpuPairLimit){
                return std::format("Dec pairs {} points; the Cpu engine runs up to {}", pairs, kCpuPairLimit);
            }
            if (terms > kCpuMsmLimit){
                return std::format("Dec runs msms of {} terms; the Cpu engine runs up to {}", terms, kCpuMsmLimit);
            }
        } else {
            const auto bytes = static_cast<double>(terms * kJacobianG1Bytes);
            const auto free = static_cast<double>(gpu_free_memory());
            if (bytes > free){
                return std::format("Dec's msm holds {:.1f} GB of terms, but {:.1f} GB of device memory is free",
                    bytes / 1e9, free / 1e9);
            }
        }
        return std::nullopt;
    }

    template <class Family, template <Engine> class Scheme, Engine E>
    void measure_scheme(const Case& c, const Sample& inputs){
        using S = Scheme<E>;
        const auto row = [&](const std::string& cells){
            std::cout << std::format("| {} | {} | {} |\n", S::name, engine_name<E>(), cells);
        };
        const auto skipped = [&](const std::string& reason){ row("skipped: " + reason + " | – | – | – | – | –"); };
        if (const auto reason = skip_reason<E>(footprint<Scheme>(c.length), c)){
            skipped(*reason);
            return;
        }

        std::optional<Timed<decltype(S::setup(c.length))>> setup;
        try{
            setup.emplace(timed([&]{ return S::setup(c.length); }));
        } catch (const DeviceError& error){
            skipped(error.what());
            return;
        }
        const auto& state = setup->result;

        const auto sk = timed([&]{ return Family::key(state, inputs); });
        const auto ct = timed([&]{ return Family::encrypt(state, inputs); });
        const auto decrypt = S::decryptor(state, Range{0, kBound});
        const auto decryption_ms = [&](const auto& key){
            const auto found = timed([&]{ return decrypt(key, ct.result); });
            if (found.result != inputs.values){
                throw std::runtime_error(std::format("{} on the {} engine decrypted wrongly", S::name, engine_name<E>()));
            }
            return found.time;
        };
        const auto dec_ms = decryption_ms(sk.result);

        std::string prepared_cells = "– | –";
        if constexpr (requires{ prepare(sk.result); }){
            const auto prepared = timed([&]{ return prepare(sk.result); });
            prepared_cells = cell(prepared.time) + " | " + cell(decryption_ms(prepared.result));
        }
        row(std::format("{} | {} | {} | {} | {}", cell(setup->time), cell(sk.time), cell(ct.time), cell(dec_ms),
            prepared_cells));
    }

    template <class Family, template <Engine> class Scheme>
    void measure_engines(const Case& c, const Sample& inputs){
        measure_scheme<Family, Scheme, Gpu>(c, inputs);
        measure_scheme<Family, Scheme, Cpu>(c, inputs);
    }

    template <class Family, template <Engine> class... Schemes>
    void table(const Case& c){
        const auto inputs = samples(c, Family::degree);
        std::cout << std::format(
            "\n## {}, n = {}, B = {} (ms per call)\n\n"
            "| Scheme | Engine | Setup | KeyGen | Enc | Dec | Prepare | Prepared Dec |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n",
            Family::title, c.length, c.batch
        );
        (measure_engines<Family, Schemes>(c, inputs), ...);
    }

    void print_header(const std::string_view mode){
        std::cout << std::format("# CudaPFE FE benchmark ({})\n\n", mode) << machine_summary() << timing_summary()
            << std::format(
                "- Each call handles a batch of B keys or ciphertexts; Dec decrypts B pairs zipped. Inputs are random "
                "vectors (and matrices) whose results lie in [0, {0}]. Fixed-base schemes reuse one discrete-log "
                "table built outside Dec; Bishop et al. and Kim et al. search the range inside Dec.\n"
                "- The Cpu engine runs on one core, for setups up to {1} x {1} matrices, decryptions up to {2} pairs "
                "and msms up to {3} terms. Quadratic FE runs up to n = {4}: past it, inputs whose quadratic forms "
                "stay within [0, {0}] have only zero entries.\n",
                kBound, kCpuMatrixLimit, kCpuPairLimit, kCpuMsmLimit, kQuadraticLengthLimit
            );
    }
}

int main(const int argc, char** argv){
    bool quick = false;
    for (int i = 1; i < argc; ++i){
        if (std::string_view(argv[i]) == "--quick") quick = true;
        else {
            std::cerr << "usage: cudapfe_bench_fe [--quick]\n";
            return 2;
        }
    }
    if (!gpu_available()){
        std::cerr << "cudapfe_bench_fe needs a CUDA device\n";
        return 1;
    }
    std::cout << std::unitbuf;
    seed(bytes_of("cudapfe_bench_fe"));
    (void)Gt::generator();

    print_header(quick ? "quick" : "full");
    for (const auto& c : quick ? kQuick : kFull){
        table<InnerProduct, Bjk, Tao, Kim, Lin, Kks, Opt>(c);
        if (c.length <= kQuadraticLengthLimit) table<Quadratic, Bcfg, Sgp>(c);
    }
}
