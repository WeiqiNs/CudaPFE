#ifndef CUFE_ENGINE_HPP
#define CUFE_ENGINE_HPP

#include <concepts>

namespace cufe{
    struct Cpu{};

    struct Gpu{};

    template <class E>
    concept Engine = std::same_as<E, Cpu> || std::same_as<E, Gpu>;

    [[nodiscard]] bool gpu_available();
}

#endif
