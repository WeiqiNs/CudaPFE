#ifndef CUFE_DLOG_HPP
#define CUFE_DLOG_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include "core.hpp"
#include "engine.hpp"
#include "vec.hpp"

namespace cufe{
    struct Range{
        std::int64_t lo;
        std::int64_t hi;
    };

    namespace detail{
        template <Engine E>
        struct BsgsTables;

        template <Engine E>
        struct BsgsBases;
    }

    template <Engine E>
    class DlogTable{
    public:
        DlogTable(const Gt& base, const Range& range);

        [[nodiscard]] std::optional<std::int64_t> find(const Gt& target) const;
        [[nodiscard]] std::vector<std::optional<std::int64_t>> find(const Vec<Gt, E>& targets) const;

    private:
        std::shared_ptr<const detail::BsgsTables<E>> tables_;
    };

    template <Engine E>
    class DlogTables{
    public:
        DlogTables(const Vec<Gt, E>& bases, const Range& range);

        [[nodiscard]] std::vector<std::optional<std::int64_t>> find(const Vec<Gt, E>& targets) const;

    private:
        std::shared_ptr<const detail::BsgsBases<E>> bases_;
    };
}

#endif
