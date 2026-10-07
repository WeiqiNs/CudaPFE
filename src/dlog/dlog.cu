#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <cub/device/device_segmented_sort.cuh>
#include <cudapfe/core.hpp>
#include <cudapfe/dlog.hpp>
#include <cudapfe/pairing.hpp>
#include <cudapfe/vec.hpp>
#include "dlog/bsgs.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "support/runtime.hpp"
#include "vec/storage.hpp"

namespace cudapfe{
    namespace detail{
        struct BsgsEntry{
            Fp12 base;
            Fp12 shift;
            Fp12 giant;
            bool trivial;
        };

        template <Engine E>
        struct BsgsTables{
            Steps steps;
            Buffer<BsgsEntry, E> entries;
            Buffer<std::uint64_t, E> fingerprints;
            Buffer<std::uint32_t, E> baby_steps;
        };

        template <Engine E>
        struct BsgsBases{
            Vec<Gt, E> bases;
            Steps steps;
        };
    }

    namespace{
        using detail::BsgsEntry;
        using detail::BsgsTables;
        using detail::Buffer;
        using detail::data;
        using detail::Fp12;
        using detail::Chunks;
        using detail::Steps;
        using detail::Word;

        constexpr Word kNotFound = std::numeric_limits<Word>::max();

        struct Targets{
            const Fp12* values;
            std::size_t count;
            Spread tables;
            Word* found;
        };

        CUDAPFE_HD std::uint64_t fingerprint(const Fp12& x){ return x.c0.c0.c0.montgomery()[0]; }

        CUDAPFE_HD Fp12 power(const Fp12& base, const std::uint64_t exponent){
            return detail::cyclotomic_pow(base, detail::Words<1>{exponent});
        }

        CUDAPFE_HD Fp12 inverse_power(const Fp12& base, const std::int64_t exponent){
            const auto magnitude = static_cast<std::uint64_t>(exponent);
            return exponent < 0 ? power(base, 0 - magnitude) : detail::conjugate(power(base, magnitude));
        }

        CUDAPFE_HD void store_min(Word* slot, const Word value){
#ifdef __CUDA_ARCH__
            atomicMin(slot, value);
#else
            *slot = std::min(*slot, value);
#endif
        }

        CUDAPFE_HD std::uint64_t lower_bound(const std::uint64_t* keys, std::uint64_t size, const std::uint64_t key){
            std::uint64_t low = 0;
            while (size > 0){
                const auto half = size / 2;
                if (keys[low + half] < key){
                    low += half + 1;
                    size -= half + 1;
                } else {
                    size = half;
                }
            }
            return low;
        }

        struct EntryOp{
            const Fp12* bases;
            Steps steps;
            BsgsEntry* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                const auto base = bases[i];
                const auto giant = detail::conjugate(power(base, steps.baby));
                out[i] = {base, inverse_power(base, steps.lo), giant, base == Fp12::one()};
            }
        };

        struct BabyStepOp{
            Chunks plan;
            const BsgsEntry* entries;
            std::uint64_t* fingerprints;
            std::uint32_t* baby_steps;

            CUDAPFE_HD void operator()(const std::size_t i) const{
                const auto ladder = plan.at(i);
                const auto& base = entries[ladder.segment].base;
                const auto offset = ladder.segment * plan.length;
                auto value = power(base, ladder.first);
#pragma unroll 1
                for (auto j = ladder.first; j < ladder.first + ladder.count; ++j){
                    fingerprints[offset + j] = fingerprint(value);
                    baby_steps[offset + j] = static_cast<std::uint32_t>(j);
                    value = value * base;
                }
            }
        };

        struct GiantStepOp{
            Chunks plan;
            Steps steps;
            Spread tables;
            const BsgsEntry* entries;
            const std::uint64_t* fingerprints;
            const std::uint32_t* baby_steps;
            const Fp12* targets;
            Word* found;

            CUDAPFE_HD void operator()(const std::size_t index) const{
                const auto ladder = plan.at(index);
                const auto table = tables == Spread::shared ? 0 : ladder.segment;
                const auto& entry = entries[table];
                if (entry.trivial){
                    if (ladder.first == 0 && targets[ladder.segment] == Fp12::one()) store_min(found + ladder.segment, 0);
                    return;
                }
                const auto* keys = fingerprints + table * steps.baby;
                const auto* values = baby_steps + table * steps.baby;
                auto gamma = targets[ladder.segment] * entry.shift * power(entry.giant, ladder.first);
#pragma unroll 1
                for (auto i = ladder.first; i < ladder.first + ladder.count; ++i){
                    const auto key = fingerprint(gamma);
#pragma unroll 1
                    for (auto at = lower_bound(keys, steps.baby, key); at < steps.baby && keys[at] == key; ++at){
                        const auto k = i * steps.baby + values[at];
                        if (k <= steps.span && power(entry.base, values[at]) == gamma){
                            store_min(found + ladder.segment, k);
                            return;
                        }
                    }
                    gamma = gamma * entry.giant;
                }
            }
        };

        template <Engine E, class Op>
        std::size_t resident_threads(){
            if constexpr (std::same_as<E, Cpu>) return 1;
            else return detail::resident_threads<Op>();
        }

        void sort_segments(BsgsTables<Cpu>& tables, const std::size_t entries){
            const auto width = tables.steps.baby;
            std::vector<std::pair<std::uint64_t, std::uint32_t>> segment(width);
            for (std::size_t e = 0; e < entries; ++e){
                const auto offset = e * width;
                for (std::size_t j = 0; j < width; ++j){
                    segment[j] = {tables.fingerprints[offset + j], tables.baby_steps[offset + j]};
                }
                std::sort(segment.begin(), segment.end());
                for (std::size_t j = 0; j < width; ++j){
                    tables.fingerprints[offset + j] = segment[j].first;
                    tables.baby_steps[offset + j] = segment[j].second;
                }
            }
        }

        void sort_segments(BsgsTables<Gpu>& tables, const std::size_t entries){
            const auto width = tables.steps.baby;
            const auto items = tables.fingerprints.size();
            std::vector<std::int64_t> bounds(entries + 1);
            for (std::size_t e = 0; e <= entries; ++e) bounds[e] = static_cast<std::int64_t>(e * width);
            const auto offsets = detail::to_engine<Gpu>(std::move(bounds));
            detail::DeviceArray<std::uint64_t> keys(items);
            detail::DeviceArray<std::uint32_t> values(items);
            const auto stream = detail::GpuRuntime::require().stream();
            const auto sort = [&](void* temp, std::size_t& temp_bytes){
                return cub::DeviceSegmentedSort::SortPairs(
                    temp, temp_bytes, tables.fingerprints.data(), keys.data(), tables.baby_steps.data(), values.data(),
                    static_cast<std::int64_t>(items), static_cast<std::int64_t>(entries), offsets.data(),
                    offsets.data() + 1, stream);
            };
            std::size_t temp_bytes = 0;
            detail::check(sort(nullptr, temp_bytes), "cub::DeviceSegmentedSort::SortPairs");
            detail::DeviceArray<std::byte> temp(std::max<std::size_t>(temp_bytes, 1));
            detail::check(sort(temp.data(), temp_bytes), "cub::DeviceSegmentedSort::SortPairs");
            tables.fingerprints = std::move(keys);
            tables.baby_steps = std::move(values);
        }

        template <Engine E>
        BsgsTables<E> build_tables(const Fp12* bases, const std::size_t count, const Steps& steps){
            Buffer<BsgsEntry, E> entries(count);
            detail::for_each<E>(count, EntryOp{bases, steps, entries.data()});
            const auto plan = detail::plan_ladders(count, steps.baby, resident_threads<E, BabyStepOp>());
            Buffer<std::uint64_t, E> fingerprints(count * steps.baby);
            Buffer<std::uint32_t, E> baby_steps(count * steps.baby);
            const BabyStepOp op{plan, entries.data(), fingerprints.data(), baby_steps.data()};
            detail::for_each<E>(plan.count(), op);
            BsgsTables<E> tables{steps, std::move(entries), std::move(fingerprints), std::move(baby_steps)};
            if (count != 0) sort_segments(tables, count);
            return tables;
        }

        template <Engine E>
        void giant_steps(const BsgsTables<E>& tables, const Targets& targets){
            const auto resident = resident_threads<E, GiantStepOp>();
            const auto plan = detail::plan_ladders(targets.count, tables.steps.giant, resident);
            detail::for_each<E>(plan.count(), GiantStepOp{
                plan, tables.steps, targets.tables, tables.entries.data(), tables.fingerprints.data(),
                tables.baby_steps.data(), targets.values, targets.found
            });
        }

        template <Engine E>
        Buffer<Word, E> not_found(const std::size_t count){
            return detail::to_engine<E>(std::vector<Word>(count, kNotFound));
        }

        std::vector<std::optional<std::int64_t>> exponents(const std::vector<Word>& found, const Steps& steps){
            std::vector<std::optional<std::int64_t>> results;
            results.reserve(found.size());
            for (const auto k : found){
                results.push_back(k == kNotFound ? std::nullopt : std::optional(detail::offset(steps.lo, k)));
            }
            return results;
        }

        template <Engine E>
        BsgsTables<E> single_table(const Gt& base, const Range& range){
            const auto steps = detail::plan_steps(range);
            const auto bases = Vec<Gt, E>::upload(std::span(&base, 1));
            auto tables = build_tables<E>(data(bases), 1, steps);
            detail::finish<E>();
            return tables;
        }
    }

    template <Engine E>
    DlogTable<E>::DlogTable(const Gt& base, const Range& range)
        : tables_(std::make_shared<const BsgsTables<E>>(single_table<E>(base, range))){}

    template <Engine E>
    std::optional<std::int64_t> DlogTable<E>::find(const Gt& target) const{
        return find(Vec<Gt, E>::upload(std::span(&target, 1))).front();
    }

    template <Engine E>
    std::vector<std::optional<std::int64_t>> DlogTable<E>::find(const Vec<Gt, E>& targets) const{
        auto found = not_found<E>(targets.size());
        giant_steps(*tables_, {data(targets), targets.size(), Spread::shared, found.data()});
        return exponents(detail::to_host(found), tables_->steps);
    }

    template <Engine E>
    DlogTables<E>::DlogTables(const Vec<Gt, E>& bases, const Range& range)
        : bases_(std::make_shared<const detail::BsgsBases<E>>(bases, detail::plan_steps(range))){}

    template <Engine E>
    std::vector<std::optional<std::int64_t>> DlogTables<E>::find(const Vec<Gt, E>& targets) const{
        const auto& [bases, steps] = *bases_;
        detail::require_same_size(bases, targets, "DlogTables::find");
        auto found = not_found<E>(targets.size());
        const auto chunk = detail::entries_per_chunk(steps.baby, detail::kTableBudgetBytes);
        for (std::size_t first = 0; first < targets.size(); first += chunk){
            const auto count = std::min(chunk, targets.size() - first);
            const auto tables = build_tables<E>(data(bases) + first, count, steps);
            giant_steps(tables, {data(targets) + first, count, Spread::per_segment, found.data() + first});
        }
        return exponents(detail::to_host(found), steps);
    }

    template class DlogTable<Cpu>;
    template class DlogTable<Gpu>;
    template class DlogTables<Cpu>;
    template class DlogTables<Gpu>;
}
