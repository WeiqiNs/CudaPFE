#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <cufe/core.hpp>
#include <cufe/linalg.hpp>
#include <cufe/vec.hpp>
#include <cufe/vector.hpp>
#include "field/field.hpp"
#include "linalg/elimination.hpp"
#include "support/access.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/storage.hpp"

namespace cufe{
    namespace{
        using detail::Buffer;
        using detail::data;
        using detail::Fr;
        using detail::generate;

        struct DiagonalOp{
            std::size_t cols;
            Fr diagonal;
            Fr* out;

            CUFE_HD void operator()(const std::size_t t) const{
                out[t] = t / cols == t % cols ? diagonal : Fr::zero();
            }
        };

        struct ProductOp{
            const Fr* a;
            const Fr* b;
            std::size_t inner;
            std::size_t cols;
            Fr* out;

            CUFE_HD void operator()(const std::size_t t) const{
                const auto row = t / cols;
                const auto col = t % cols;
                auto total = Fr::zero();
#pragma unroll 1
                for (std::size_t k = 0; k < inner; ++k) total = total + a[row * inner + k] * b[k * cols + col];
                out[t] = total;
            }
        };

        struct TransposeOp{
            const Fr* in;
            Shape shape;
            Fr* out;

            CUFE_HD void operator()(const std::size_t t) const{
                out[t] = in[t % shape.rows * shape.cols + t / shape.rows];
            }
        };

        struct ScaleOp{
            const Fr* in;
            Fr k;
            Fr* out;

            CUFE_HD void operator()(const std::size_t t) const{ out[t] = in[t] * k; }
        };

        struct EqualOp{
            const Fr* x;
            const Fr* y;
            unsigned* differs;

            CUFE_HD void operator()(const std::size_t t) const{
                if (!(x[t] == y[t])) *differs = 1;
            }
        };

        template <Engine E>
        struct Elimination{
            Zp determinant;
            std::optional<Vec<Zp, E>> inverse;
        };

        std::size_t entry_count(const Shape& shape){
            if (shape.rows == 0 || shape.cols == 0) throw ShapeError("a matrix needs at least one row and one column");
            if (shape.cols > std::numeric_limits<std::size_t>::max() / shape.rows){
                throw ShapeError("a matrix of " + std::to_string(shape.rows) + " x " + std::to_string(shape.cols)
                    + " entries overflows");
            }
            return shape.rows * shape.cols;
        }

        void require_square(const Shape& shape){
            if (shape.rows != shape.cols) throw ShapeError("only square matrices have a determinant or an inverse");
        }

        template <Engine E>
        Elimination<E> eliminate(const Vec<Zp, E>& matrix, const std::size_t m){
            Buffer<Fr, E> work(2 * m * m);
            const detail::Workspace workspace{work.data(), m};
            detail::for_each<E>(work.size(), detail::AugmentOp{data(matrix), workspace});

            auto state = detail::to_engine<E>(std::vector{detail::EliminationState{Fr::one(), Fr::zero(), false}});
            std::vector<std::size_t> rows(m);
            std::iota(rows.begin(), rows.end(), std::size_t{0});
            auto pivots = detail::to_engine<E>(std::move(rows));
            Buffer<Fr, E> factors(m);
            for (std::size_t c = 0; c < m; ++c){
                detail::for_each<E>(1, detail::PivotOp{workspace, c, state.data(), pivots.data()});
                detail::for_each<E>(m + 1, detail::ScaleRowOp{workspace, c, state.data()});
                detail::for_each<E>(m, detail::SaveColumnOp{workspace, c, factors.data()});
                detail::for_each<E>(m * (m + 1), detail::EliminateOp{workspace, c, factors.data()});
            }

            Buffer<std::size_t, E> columns(m);
            detail::for_each<E>(1, detail::ColumnOrderOp{pivots.data(), m, columns.data()});
            Buffer<Fr, E> inverse(m * m);
            detail::for_each<E>(inverse.size(), detail::ExtractOp{work.data(), m, columns.data(), inverse.data()});

            const auto result = detail::element_at(state, 0);
            const auto determinant = detail::from_element<Zp>(result.determinant);
            if (result.singular) return {determinant, std::nullopt};
            return {determinant, detail::vec<Zp, E>(std::move(inverse))};
        }
    }

    template <Engine E>
    Matrix<E>::Matrix(const Shape& shape, Vec<Zp, E> entries) : shape_(shape), entries_(std::move(entries)){}

    template <Engine E>
    Matrix<E> Matrix<E>::zeros(const Shape& shape){
        const auto count = entry_count(shape);
        return Matrix(shape, generate<Zp, E>(count, [&](auto* out){ return DiagonalOp{shape.cols, Fr::zero(), out}; }));
    }

    template <Engine E>
    Matrix<E> Matrix<E>::upload(const Shape& shape, const std::span<const Zp> row_major){
        const auto count = entry_count(shape);
        if (row_major.size() != count){
            throw ShapeError("a " + std::to_string(shape.rows) + " x " + std::to_string(shape.cols) + " matrix needs "
                + std::to_string(count) + " entries, got " + std::to_string(row_major.size()));
        }
        return Matrix(shape, Vec<Zp, E>::upload(row_major));
    }

    template <Engine E>
    Matrix<E> Matrix<E>::from_rows(const std::vector<Vector>& rows){
        const Shape shape{rows.size(), rows.empty() ? 0 : rows.front().size()};
        Vector entries;
        entries.reserve(entry_count(shape));
        for (const auto& row : rows){
            if (row.size() != shape.cols) throw ShapeError("matrix rows must all have the same length");
            entries.insert(entries.end(), row.begin(), row.end());
        }
        return upload(shape, entries);
    }

    template <Engine E>
    Matrix<E> Matrix<E>::identity(const std::size_t size){
        const Shape shape{size, size};
        const auto count = entry_count(shape);
        return Matrix(shape, generate<Zp, E>(count, [&](auto* out){ return DiagonalOp{size, Fr::one(), out}; }));
    }

    template <Engine E>
    Matrix<E> Matrix<E>::random(const Shape& shape){
        return upload(shape, random_vector(entry_count(shape)));
    }

    template <Engine E>
    std::vector<Vector> Matrix<E>::to_rows() const{
        const auto entries = entries_.download();
        std::vector<Vector> rows;
        rows.reserve(shape_.rows);
        for (auto row = entries.begin(); row != entries.end(); row += static_cast<std::ptrdiff_t>(shape_.cols)){
            rows.emplace_back(row, row + static_cast<std::ptrdiff_t>(shape_.cols));
        }
        return rows;
    }

    template <Engine E>
    Matrix<E> Matrix<E>::transpose() const{
        return Matrix({shape_.cols, shape_.rows}, generate<Zp, E>(entries_.size(), [&](auto* out){
            return TransposeOp{data(entries_), shape_, out};
        }));
    }

    template <Engine E>
    bool Matrix<E>::is_identity() const{
        return shape_.rows == shape_.cols && *this == identity(shape_.rows);
    }

    template <Engine E>
    Zp Matrix<E>::determinant() const{
        require_square(shape_);
        return eliminate(entries_, shape_.rows).determinant;
    }

    template <Engine E>
    Matrix<E> Matrix<E>::inverse() const{
        return inverse_with_determinant().inverse;
    }

    template <Engine E>
    Inversion<E> Matrix<E>::inverse_with_determinant() const{
        require_square(shape_);
        auto [determinant, inverse] = eliminate(entries_, shape_.rows);
        if (!inverse) throw NotInvertible("a singular matrix has no inverse");
        return {Matrix(shape_, std::move(*inverse)), determinant};
    }

    template <Engine E>
    bool Matrix<E>::equals(const Matrix& b) const{
        if (shape_ != b.shape_) return false;
        auto differs = detail::to_engine<E>(std::vector<unsigned>{0});
        detail::for_each<E>(entries_.size(), EqualOp{data(entries_), data(b.entries_), differs.data()});
        return detail::element_at(differs, 0) == 0;
    }

    template <Engine E>
    Matrix<E> Matrix<E>::times(const Matrix& b) const{
        if (shape_.cols != b.shape_.rows) throw ShapeError("matrix product needs left columns equal to right rows");
        const Shape shape{shape_.rows, b.shape_.cols};
        return Matrix(shape, generate<Zp, E>(entry_count(shape), [&](auto* out){
            return ProductOp{data(entries_), data(b.entries_), shape_.cols, shape.cols, out};
        }));
    }

    template <Engine E>
    Matrix<E> Matrix<E>::scaled(const Zp& k) const{
        const auto factor = detail::Access::element<Fr>(k);
        return Matrix(shape_, generate<Zp, E>(entries_.size(), [&](auto* out){
            return ScaleOp{data(entries_), factor, out};
        }));
    }

    template class Matrix<Cpu>;
    template class Matrix<Gpu>;
}
