#ifndef CUFE_VEC_HPP
#define CUFE_VEC_HPP

#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include "core.hpp"
#include "engine.hpp"

namespace cufe{
    namespace detail{
        template <class T, Engine E>
        struct Storage;

        template <class T>
        concept GroupPoint = std::same_as<T, G1> || std::same_as<T, G2>;
    }

    template <class T, Engine E>
    class Vec{
    public:
        Vec();

        [[nodiscard]] static Vec upload(std::span<const T> values);

        [[nodiscard]] std::vector<T> download() const;
        [[nodiscard]] T at(std::size_t index) const;
        [[nodiscard]] std::size_t size() const;
        [[nodiscard]] bool empty() const{ return size() == 0; }

        friend Vec operator+(const Vec& x, const Vec& y) requires detail::GroupPoint<T>{ return x.plus(y); }
        friend Vec operator-(const Vec& x, const Vec& y) requires detail::GroupPoint<T>{ return x.plus(y.negated()); }
        friend Vec operator-(const Vec& x) requires detail::GroupPoint<T>{ return x.negated(); }

        friend Vec operator*(const Vec& x, const Vec<Zp, E>& k) requires detail::GroupPoint<T>{ return x.scaled(k); }

        friend Vec operator*(const Vec& x, const Vec& y) requires std::same_as<T, Gt>{ return x.times(y); }
        friend Vec operator/(const Vec& x, const Vec& y) requires std::same_as<T, Gt>{ return x.divided(y); }

    private:
        friend struct detail::Access;

        explicit Vec(std::shared_ptr<const detail::Storage<T, E>> storage);

        [[nodiscard]] Vec plus(const Vec& y) const requires detail::GroupPoint<T>;
        [[nodiscard]] Vec negated() const requires detail::GroupPoint<T>;
        [[nodiscard]] Vec scaled(const Vec<Zp, E>& k) const requires detail::GroupPoint<T>;
        [[nodiscard]] Vec times(const Vec& y) const requires std::same_as<T, Gt>;
        [[nodiscard]] Vec divided(const Vec& y) const requires std::same_as<T, Gt>;

        std::shared_ptr<const detail::Storage<T, E>> storage_;
    };

    template <class G, Engine E> requires detail::GroupPoint<G>
    [[nodiscard]] Vec<G, E> mul_generator(const Vec<Zp, E>& scalars);
}

#endif
