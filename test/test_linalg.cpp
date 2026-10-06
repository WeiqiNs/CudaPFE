#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>
#include <support/engines.hpp>

using namespace cudapfe;

namespace{
    template <Engine E>
    Matrix<E> matrix(const std::vector<std::vector<int>>& rows){
        std::vector<Vector> converted;
        for (const auto& row : rows) converted.emplace_back(row.begin(), row.end());
        return Matrix<E>::from_rows(converted);
    }

    template <Engine E>
    void expect_inverse_pair(const Matrix<E>& a, const Matrix<E>& inverse){
        EXPECT_TRUE((a * inverse).is_identity());
        EXPECT_TRUE((inverse * a).is_identity());
    }
}

template <class E>
class MatrixTest : public EngineTest<E>{};

TYPED_TEST_SUITE(MatrixTest, Engines);

TYPED_TEST(MatrixTest, ProductsAndTranspose){
    using M = Matrix<TypeParam>;
    const auto a = matrix<TypeParam>({{1, 2}, {3, 4}});
    const auto wide = matrix<TypeParam>({{1, 2, 3}, {4, 5, 6}});

    EXPECT_EQ(a * matrix<TypeParam>({{0, 1}, {1, 0}}), matrix<TypeParam>({{2, 1}, {4, 3}}));
    EXPECT_EQ(matrix<TypeParam>({{5, 6}}) * a, matrix<TypeParam>({{23, 34}}));
    EXPECT_EQ(a * matrix<TypeParam>({{5}, {6}}), matrix<TypeParam>({{17}, {39}}));
    EXPECT_EQ(matrix<TypeParam>({{1, 2}}) * wide, matrix<TypeParam>({{9, 12, 15}}));
    EXPECT_EQ(a * Zp(2), matrix<TypeParam>({{2, 4}, {6, 8}}));
    EXPECT_EQ(Zp(3) * a, matrix<TypeParam>({{3, 6}, {9, 12}}));
    EXPECT_EQ(wide.transpose(), matrix<TypeParam>({{1, 4}, {2, 5}, {3, 6}}));
    EXPECT_EQ(wide.shape(), (Shape{2, 3}));
    EXPECT_EQ(wide.transpose().shape(), (Shape{3, 2}));

    const std::vector<Zp> entries{1, 2, 3, 4, 5, 6};
    EXPECT_EQ(M::upload({2, 3}, entries), wide);
    EXPECT_EQ(wide.entries().download(), entries);
    EXPECT_EQ(wide.to_rows(), (std::vector<Vector>{{1, 2, 3}, {4, 5, 6}}));
    EXPECT_NE(M::upload({3, 2}, entries), wide);
    EXPECT_EQ(M::zeros({2, 3}), matrix<TypeParam>({{0, 0, 0}, {0, 0, 0}}));
    EXPECT_EQ(M::identity(3), matrix<TypeParam>({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}));
    EXPECT_TRUE(M::identity(3).is_identity());
    EXPECT_FALSE(matrix<TypeParam>({{1, 0}, {0, 2}}).is_identity());
    EXPECT_FALSE(matrix<TypeParam>({{1, 0}}).is_identity());
}

TYPED_TEST(MatrixTest, InverseAndDeterminantUseRowSwaps){
    const auto swap = matrix<TypeParam>({{0, 1}, {1, 0}});
    const auto [swap_inverse, swap_determinant] = swap.inverse_with_determinant();
    EXPECT_EQ(swap_inverse, swap);
    EXPECT_EQ(swap_determinant, Zp(-1));

    EXPECT_EQ(matrix<TypeParam>({{2, 1}, {1, 1}}).inverse(), matrix<TypeParam>({{1, -1}, {-1, 2}}));

    const auto three = matrix<TypeParam>({{3, 5, 8}, {2, 2, 2}, {9, 9, 3}});
    EXPECT_EQ(three.determinant(), Zp(24));
    expect_inverse_pair(three, three.inverse());

    const auto late_swap = matrix<TypeParam>({{1, 1, 0}, {1, 1, 1}, {0, 1, 1}});
    const auto [late_inverse, late_determinant] = late_swap.inverse_with_determinant();
    EXPECT_EQ(late_inverse, matrix<TypeParam>({{0, 1, -1}, {1, -1, 1}, {-1, 1, 0}}));
    EXPECT_EQ(late_determinant, Zp(-1));

    const auto cycle = matrix<TypeParam>({{0, 0, 0, 2}, {3, 0, 0, 0}, {0, 5, 0, 0}, {0, 0, 7, 0}});
    const auto [cycle_inverse, cycle_determinant] = cycle.inverse_with_determinant();
    const std::vector<Vector> cycle_rows{
        {0, Zp(3).inverse(), 0, 0}, {0, 0, Zp(5).inverse(), 0}, {0, 0, 0, Zp(7).inverse()}, {Zp(2).inverse(), 0, 0, 0},
    };
    EXPECT_EQ(cycle_inverse, Matrix<TypeParam>::from_rows(cycle_rows));
    EXPECT_EQ(cycle_determinant, Zp(-210));
}

TYPED_TEST(MatrixTest, SingularAndNonSquareAreRejected){
    using M = Matrix<TypeParam>;
    const auto late_singular = matrix<TypeParam>({{0, 1, 1}, {1, 0, 0}, {1, 1, 1}});
    for (const auto& singular : {matrix<TypeParam>({{1, 2}, {2, 4}}), late_singular}){
        EXPECT_TRUE(singular.determinant().is_zero());
        EXPECT_THROW((void)singular.inverse(), NotInvertible);
        EXPECT_THROW((void)singular.inverse_with_determinant(), NotInvertible);
    }

    const auto wide = matrix<TypeParam>({{1, 2, 3}, {4, 5, 6}});
    EXPECT_THROW((void)wide.determinant(), ShapeError);
    EXPECT_THROW((void)wide.inverse(), ShapeError);
    EXPECT_THROW((void)(wide * wide), ShapeError);
    EXPECT_THROW((void)M::zeros({0, 2}), ShapeError);
    EXPECT_THROW((void)M::zeros({2, 0}), ShapeError);
    EXPECT_THROW((void)M::identity(0), ShapeError);
    EXPECT_THROW((void)M::random({0, 1}), ShapeError);
    EXPECT_THROW((void)M::from_rows({}), ShapeError);
    EXPECT_THROW((void)M::from_rows({Vector(2), Vector(3)}), ShapeError);
    EXPECT_THROW((void)M::upload({2, 2}, std::vector<Zp>(3)), ShapeError);
}

TYPED_TEST(MatrixTest, RandomMatricesInvert){
    for (const auto m : {std::size_t{1}, std::size_t{17}, std::size_t{130}}){
        const auto a = Matrix<TypeParam>::random({m, m});
        const auto [inverse, determinant] = a.inverse_with_determinant();
        expect_inverse_pair(a, inverse);
        EXPECT_EQ(determinant * inverse.determinant(), Zp(1)) << m;
    }
}
