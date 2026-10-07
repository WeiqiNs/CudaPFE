#ifndef CUDAPFE_TEST_ENGINES_HPP
#define CUDAPFE_TEST_ENGINES_HPP

#include <concepts>
#include <gtest/gtest.h>
#include <cudapfe/engine.hpp>

#define CUDAPFE_REQUIRE_GPU() \
    do{ \
        if (!::cudapfe::gpu_available()) GTEST_SKIP() << "no usable CUDA device"; \
    } while (false)

using Engines = ::testing::Types<cudapfe::Cpu, cudapfe::Gpu>;

template <cudapfe::Engine E>
class EngineTest : public ::testing::Test{
protected:
    void SetUp() override{
        if constexpr (std::same_as<E, cudapfe::Gpu>) CUDAPFE_REQUIRE_GPU();
    }
};

template <class T, cudapfe::Engine E>
struct Case{
    using Value = T;
    using Engine = E;
};

template <class C>
class CaseTest : public EngineTest<typename C::Engine>{};

#endif
