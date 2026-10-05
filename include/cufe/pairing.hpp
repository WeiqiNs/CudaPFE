#ifndef CUFE_PAIRING_HPP
#define CUFE_PAIRING_HPP

#include <cstddef>
#include <memory>
#include "core.hpp"
#include "engine.hpp"
#include "vec.hpp"

namespace cufe{
    struct PairShape{
        std::size_t segments;
        std::size_t length;
        Spread p = Spread::per_segment;
        Spread q = Spread::per_segment;
    };

    namespace detail{
        template <Engine E>
        struct LineStorage;
    }

    template <Engine E>
    class G2Lines{
    public:
        [[nodiscard]] std::size_t size() const;

    private:
        friend struct detail::Access;

        explicit G2Lines(std::shared_ptr<const detail::LineStorage<E>> storage);

        std::shared_ptr<const detail::LineStorage<E>> storage_;
    };

    template <Engine E>
    [[nodiscard]] G2Lines<E> prepare(const Vec<G2, E>& qs);

    template <Engine E>
    [[nodiscard]] Vec<Gt, E> pair_segments(const Vec<G1, E>& ps, const G2Lines<E>& qs, const PairShape& shape);

    template <Engine E>
    [[nodiscard]] Vec<Gt, E> pair_segments(const Vec<G1, E>& ps, const Vec<G2, E>& qs, const PairShape& shape);
}

#endif
