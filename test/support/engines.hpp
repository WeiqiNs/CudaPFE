#ifndef CUFE_TEST_ENGINES_HPP
#define CUFE_TEST_ENGINES_HPP

#include <concepts>
#include <gtest/gtest.h>
#include <cufe/engine.hpp>

#define CUFE_REQUIRE_GPU() \
    do{ \
        if (!::cufe::gpu_available()) GTEST_SKIP() << "no CUDA device"; \
    } while (false)

using Engines = ::testing::Types<cufe::Cpu, cufe::Gpu>;

template <cufe::Engine E>
class EngineTest : public ::testing::Test{
protected:
    void SetUp() override{
        if constexpr (std::same_as<E, cufe::Gpu>) CUFE_REQUIRE_GPU();
    }
};

#endif
