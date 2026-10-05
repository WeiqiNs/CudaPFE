#ifndef CUFE_TEST_FE_SCHEME_TYPES_HPP
#define CUFE_TEST_FE_SCHEME_TYPES_HPP

#include <tuple>
#include <utility>
#include <gtest/gtest.h>
#include <support/engines.hpp>
#include "schemes.hpp"

template <class Tuple>
struct AsTypes;

template <class... Ts>
struct AsTypes<std::tuple<Ts...>>{
    using type = ::testing::Types<Ts...>;
};

template <template <cufe::Engine> class Scheme, cufe::Engine... Es>
using Instances = std::tuple<Scheme<Es>...>;

template <template <cufe::Engine> class... Schemes>
struct OnEveryEngine{
    using type = typename AsTypes<
        decltype(std::tuple_cat(std::declval<Instances<Schemes, cufe::Cpu, cufe::Gpu>>()...))>::type;
};

template <class Scheme>
class SchemeTest : public EngineTest<typename Scheme::Engine>{};

using InnerProductSchemes = OnEveryEngine<Bjk, Tao, Kim, Lin, Kks, Opt>::type;

#endif
