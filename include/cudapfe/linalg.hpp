#ifndef CUDAPFE_LINALG_HPP
#define CUDAPFE_LINALG_HPP

#include <cstddef>
#include <span>
#include <vector>
#include "core.hpp"
#include "engine.hpp"
#include "vec.hpp"
#include "vector.hpp"

namespace cudapfe{
    template <Engine E>
    class Matrix;

    template <Engine E>
    struct Inversion{
        Matrix<E> inverse;
        Zp determinant;
    };

    template <Engine E>
    class Matrix{
    public:
        [[nodiscard]] static Matrix zeros(const Shape& shape);
        [[nodiscard]] static Matrix upload(const Shape& shape, std::span<const Zp> row_major);
        [[nodiscard]] static Matrix from_rows(const std::vector<Vector>& rows);
        [[nodiscard]] static Matrix identity(std::size_t size);
        [[nodiscard]] static Matrix random(const Shape& shape);

        [[nodiscard]] Shape shape() const{ return shape_; }
        [[nodiscard]] const Vec<Zp, E>& entries() const{ return entries_; }
        [[nodiscard]] std::vector<Vector> to_rows() const;

        [[nodiscard]] Matrix transpose() const;
        [[nodiscard]] bool is_identity() const;
        [[nodiscard]] Zp determinant() const;
        [[nodiscard]] Matrix inverse() const;
        [[nodiscard]] Inversion<E> inverse_with_determinant() const;

        friend bool operator==(const Matrix& a, const Matrix& b){ return a.equals(b); }
        friend Matrix operator*(const Matrix& a, const Matrix& b){ return a.times(b); }
        friend Matrix operator*(const Matrix& a, const Zp& k){ return a.scaled(k); }
        friend Matrix operator*(const Zp& k, const Matrix& a){ return a.scaled(k); }

    private:
        Matrix(const Shape& shape, Vec<Zp, E> entries);

        [[nodiscard]] bool equals(const Matrix& b) const;
        [[nodiscard]] Matrix times(const Matrix& b) const;
        [[nodiscard]] Matrix scaled(const Zp& k) const;

        Shape shape_;
        Vec<Zp, E> entries_;
    };
}

#endif
