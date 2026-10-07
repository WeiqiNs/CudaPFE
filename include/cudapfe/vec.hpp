#ifndef CUDAPFE_VEC_HPP
#define CUDAPFE_VEC_HPP

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>
#include "core.hpp"
#include "engine.hpp"

namespace cudapfe{
    struct Shape{
        std::size_t rows;
        std::size_t cols;

        friend bool operator==(const Shape& x, const Shape& y) = default;
    };

    enum class Spread{ per_segment, shared };

    struct MsmShape{
        std::size_t segments;
        Shape shape;
        Spread bases = Spread::per_segment;
        Spread scalars = Spread::per_segment;
    };

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

        template <Engine To>
        [[nodiscard]] Vec<T, To> to() const{
            if constexpr (std::same_as<To, E>) return *this;
            else return transferred();
        }

        friend Vec operator+(const Vec& x, const Vec& y) requires detail::GroupPoint<T>{ return x.plus(y); }
        friend Vec operator-(const Vec& x, const Vec& y) requires detail::GroupPoint<T>{ return x.plus(y.negated()); }
        friend Vec operator-(const Vec& x) requires detail::GroupPoint<T>{ return x.negated(); }

        friend Vec operator*(const Vec& x, const Vec<Zp, E>& k) requires detail::GroupPoint<T>{ return x.scaled(k); }

        friend Vec operator*(const Vec& x, const Vec& y) requires std::same_as<T, Gt>{ return x.times(y); }
        friend Vec operator/(const Vec& x, const Vec& y) requires std::same_as<T, Gt>{ return x.divided(y); }

    private:
        friend struct detail::Access;

        explicit Vec(std::shared_ptr<const detail::Storage<T, E>> storage);

        [[nodiscard]] Vec<T, std::conditional_t<std::same_as<E, Cpu>, Gpu, Cpu>> transferred() const;

        [[nodiscard]] Vec plus(const Vec& y) const requires detail::GroupPoint<T>;
        [[nodiscard]] Vec negated() const requires detail::GroupPoint<T>;
        [[nodiscard]] Vec scaled(const Vec<Zp, E>& k) const requires detail::GroupPoint<T>;
        [[nodiscard]] Vec times(const Vec& y) const requires std::same_as<T, Gt>;
        [[nodiscard]] Vec divided(const Vec& y) const requires std::same_as<T, Gt>;

        std::shared_ptr<const detail::Storage<T, E>> storage_;
    };

    template <class G, Engine E> requires detail::GroupPoint<G>
    [[nodiscard]] Vec<G, E> mul_generator(const Vec<Zp, E>& scalars);

    template <class T, Engine E>
    struct Segments{
        Vec<T, E> values;
        std::size_t length;
        Spread spread = Spread::per_segment;
    };

    template <class G, Engine E> requires detail::GroupPoint<G>
    [[nodiscard]] Vec<G, E> msm(const Vec<G, E>& bases, const Vec<Zp, E>& scalars, const MsmShape& shape);

    template <class G, Engine E> requires detail::GroupPoint<G>
    [[nodiscard]] Vec<G, E> concat(std::size_t segments, std::initializer_list<Segments<G, E>> parts);
}

#endif
