#ifndef CUDAPFE_ENGINE_HPP
#define CUDAPFE_ENGINE_HPP

#include <concepts>
#include <cstddef>

namespace cudapfe{
    struct Cpu{};

    struct Gpu{};

    template <class E>
    concept Engine = std::same_as<E, Cpu> || std::same_as<E, Gpu>;

    [[nodiscard]] bool gpu_available();

    [[nodiscard]] std::size_t gpu_free_memory();
}

#endif
