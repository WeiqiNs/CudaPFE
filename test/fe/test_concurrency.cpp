#include <cstddef>
#include <cstdint>
#include <latch>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "scheme_types.hpp"

using cudapfe::IPFE::IntMatrix;
using cudapfe::IPFE::IntVec;

namespace{
    constexpr std::size_t kThreads = 4;
    constexpr std::size_t kBatch = 3;
    constexpr std::int64_t kEntryBound = 3;

    struct Decryptions{
        Results plain;
        Results prepared;
    };

    template <class Work>
    auto on_threads(const Work& work){
        std::vector<decltype(work(std::size_t{0}))> results(kThreads);
        std::latch start(kThreads);
        {
            std::vector<std::jthread> threads;
            for (std::size_t t = 0; t < kThreads; ++t){
                threads.emplace_back([&, t]{
                    start.arrive_and_wait();
                    results[t] = work(t);
                });
            }
        }
        return results;
    }

    std::int64_t inner_product(const IntVec& x, const IntVec& y){
        std::int64_t total = 0;
        for (std::size_t i = 0; i < x.size(); ++i) total += x[i] * y[i];
        return total;
    }

    std::int64_t quadratic_form(const IntVec& x, const IntMatrix& f, const IntVec& y){
        std::int64_t total = 0;
        for (std::size_t i = 0; i < x.size(); ++i) total += x[i] * inner_product(f[i], y);
        return total;
    }
}

template <class Scheme>
class ConcurrentInnerProductTest : public CaseTest<Scheme>{};

TYPED_TEST_SUITE(ConcurrentInnerProductTest, InnerProductSchemes);

TYPED_TEST(ConcurrentInnerProductTest, ThreadsEncryptAndDecryptUnderOneSharedKey){
    const auto msk = TypeParam::setup(4);
    const auto decrypt = TypeParam::decryptor(msk, kRange);
    const IntVec function{2, -1, 3, 1};
    const auto sk = keygen(msk, function);
    const auto prepared = prepare(sk);
    std::vector<IntMatrix> messages;
    std::vector<Results> expected(kThreads);
    for (std::size_t t = 0; t < kThreads; ++t){
        messages.push_back(random_rows(kBatch, 4, kEntryBound));
        for (const auto& message : messages.back()) expected[t].push_back(inner_product(function, message));
    }

    const auto results = on_threads([&](const std::size_t t){
        const auto ct = enc(msk, messages[t]);
        return Decryptions{decrypt(sk, ct), decrypt(prepared, ct)};
    });

    for (std::size_t t = 0; t < kThreads; ++t){
        EXPECT_EQ(results[t].plain, expected[t]) << "thread " << t;
        EXPECT_EQ(results[t].prepared, expected[t]) << "thread " << t;
    }
}

template <class Scheme>
class ConcurrentQuadraticTest : public CaseTest<Scheme>{};

TYPED_TEST_SUITE(ConcurrentQuadraticTest, QuadraticSchemes);

TYPED_TEST(ConcurrentQuadraticTest, ThreadsEncryptAndDecryptUnderOneSharedKey){
    const auto keys = TypeParam::setup(3);
    const auto decrypt = TypeParam::decryptor(keys, kRange);
    const IntMatrix function{{1, 0, 2}, {0, -1, 0}, {3, 1, 1}};
    const auto sk = keygen(keys.msk, function);
    std::vector<IntMatrix> lefts;
    std::vector<IntMatrix> rights;
    std::vector<Results> expected(kThreads);
    for (std::size_t t = 0; t < kThreads; ++t){
        lefts.push_back(random_rows(kBatch, 3, kEntryBound));
        rights.push_back(random_rows(kBatch, 3, kEntryBound));
        for (std::size_t i = 0; i < kBatch; ++i) expected[t].push_back(quadratic_form(lefts[t][i], function, rights[t][i]));
    }

    const auto results = on_threads([&](const std::size_t t){ return decrypt(sk, enc(keys.pk, lefts[t], rights[t])); });

    for (std::size_t t = 0; t < kThreads; ++t) EXPECT_EQ(results[t], expected[t]) << "thread " << t;
}
