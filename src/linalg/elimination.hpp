#ifndef CUDAPFE_LINALG_ELIMINATION_HPP
#define CUDAPFE_LINALG_ELIMINATION_HPP

#include <cstddef>
#include "field/field.hpp"
#include "support/hd.hpp"

namespace cudapfe::detail{
    struct EliminationState{
        Fr determinant;
        Fr inverse_pivot;
        bool singular;
    };

    struct Workspace{
        Fr* entries;
        std::size_t size;

        [[nodiscard]] CUDAPFE_HD Fr& at(const std::size_t row, const std::size_t col) const{
            return entries[row * 2 * size + col];
        }
    };

    struct AugmentOp{
        const Fr* matrix;
        Workspace work;

        CUDAPFE_HD void operator()(const std::size_t t) const{
            const auto m = work.size;
            const auto row = t / (2 * m);
            const auto col = t % (2 * m);
            if (col < m) work.at(row, col) = matrix[row * m + col];
            else work.at(row, col) = col - m == row ? Fr::one() : Fr::zero();
        }
    };

    struct PivotOp{
        Workspace work;
        std::size_t column;
        EliminationState* state;
        std::size_t* pivots;

        CUDAPFE_HD void operator()(std::size_t) const{
            if (state->singular) return;
            const auto m = work.size;
            auto row = column;
            while (row < m && work.at(row, column).is_zero()) ++row;
            if (row == m){
                state->singular = true;
                state->determinant = Fr::zero();
                return;
            }
            pivots[column] = row;
            if (row != column){
#pragma unroll 1
                for (auto col = column; col < m + column; ++col){
                    const auto held = work.at(column, col);
                    work.at(column, col) = work.at(row, col);
                    work.at(row, col) = held;
                }
                state->determinant = -state->determinant;
            }
            const auto pivot = work.at(column, column);
            state->determinant = state->determinant * pivot;
            state->inverse_pivot = pivot.inverse();
        }
    };

    struct ScaleRowOp{
        Workspace work;
        std::size_t column;
        const EliminationState* state;

        CUDAPFE_HD void operator()(const std::size_t i) const{
            auto& entry = work.at(column, column + i);
            entry = entry * state->inverse_pivot;
        }
    };

    struct SaveColumnOp{
        Workspace work;
        std::size_t column;
        Fr* factors;

        CUDAPFE_HD void operator()(const std::size_t row) const{ factors[row] = work.at(row, column); }
    };

    struct EliminateOp{
        Workspace work;
        std::size_t column;
        const Fr* factors;

        CUDAPFE_HD void operator()(const std::size_t t) const{
            const auto width = work.size + 1;
            const auto row = t / width;
            if (row == column) return;
            const auto col = column + t % width;
            work.at(row, col) = work.at(row, col) - factors[row] * work.at(column, col);
        }
    };

    struct ColumnOrderOp{
        const std::size_t* pivots;
        std::size_t size;
        std::size_t* columns;

        CUDAPFE_HD void operator()(std::size_t) const{
#pragma unroll 1
            for (std::size_t j = 0; j < size; ++j) columns[j] = j;
#pragma unroll 1
            for (auto c = size; c-- > 0;){
                const auto held = columns[c];
                columns[c] = columns[pivots[c]];
                columns[pivots[c]] = held;
            }
        }
    };

    struct ExtractOp{
        const Fr* work;
        std::size_t size;
        const std::size_t* columns;
        Fr* out;

        CUDAPFE_HD void operator()(const std::size_t t) const{
            const auto row = t / size;
            out[t] = work[row * 2 * size + size + columns[t % size]];
        }
    };
}

#endif
