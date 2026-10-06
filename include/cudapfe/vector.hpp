#ifndef CUDAPFE_VECTOR_HPP
#define CUDAPFE_VECTOR_HPP

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>
#include "core.hpp"

namespace cudapfe{
    using Vector = std::vector<Zp>;

    namespace detail{
        template <class F>
        [[nodiscard]] Vector elementwise(const Vector& x, const Vector& y, const char* operation, const F f){
            if (x.size() != y.size()) throw ShapeError(std::string(operation) + " needs vectors of equal length");
            Vector r;
            r.reserve(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) r.push_back(f(x[i], y[i]));
            return r;
        }
    }

    [[nodiscard]] inline Vector random_vector(const std::size_t size){
        Vector r;
        r.reserve(size);
        for (std::size_t i = 0; i < size; ++i) r.push_back(Zp::random());
        return r;
    }

    [[nodiscard]] inline Vector operator+(const Vector& x, const Vector& y){
        return detail::elementwise(x, y, "vector addition", std::plus<>{});
    }

    [[nodiscard]] inline Vector operator-(const Vector& x, const Vector& y){
        return detail::elementwise(x, y, "vector subtraction", std::minus<>{});
    }

    [[nodiscard]] inline Vector operator*(const Vector& x, const Zp& k){
        Vector r;
        r.reserve(x.size());
        for (const auto& v : x) r.push_back(v * k);
        return r;
    }

    [[nodiscard]] inline Vector operator*(const Zp& k, const Vector& x){
        return x * k;
    }

    [[nodiscard]] inline Vector hadamard(const Vector& x, const Vector& y){
        return detail::elementwise(x, y, "hadamard", std::multiplies<>{});
    }

    [[nodiscard]] inline Zp sum(const Vector& x){
        Zp total;
        for (const auto& v : x) total += v;
        return total;
    }

    [[nodiscard]] inline Zp inner(const Vector& x, const Vector& y){
        return sum(hadamard(x, y));
    }

    [[nodiscard]] inline Vector concat(const std::initializer_list<Vector> parts){
        Vector r;
        for (const auto& part : parts) r.insert(r.end(), part.begin(), part.end());
        return r;
    }
}

#endif
