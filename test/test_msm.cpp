#include <cstddef>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <cufe/cufe.hpp>
#include <support/engines.hpp>

using namespace cufe;

namespace{
    struct ShapeMismatch{
        std::string_view name;
        Shape shape;
        std::size_t bases;
        std::size_t scalars;
    };

    std::vector<Zp> scalars_with_zeros(const std::size_t count){
        std::vector<Zp> scalars;
        for (std::size_t i = 0; i < count; ++i){
            scalars.push_back(i % 7 == 2 ? Zp() : i % 7 == 5 ? Zp(-1) : Zp::random());
        }
        return scalars;
    }

    template <class P>
    std::vector<P> exponent_columns(const std::vector<Zp>& logs, const std::vector<Zp>& f, const Shape& shape){
        std::vector<P> columns;
        for (std::size_t j = 0; j < shape.cols; ++j){
            Zp exponent;
            for (std::size_t i = 0; i < shape.rows; ++i) exponent += logs[i] * f[i * shape.cols + j];
            columns.push_back(P::mul_generator(exponent));
        }
        return columns;
    }
}

template <class C>
class MsmTest : public CaseTest<C>{};

using MsmCases = ::testing::Types<Case<G1, Cpu>, Case<G1, Gpu>, Case<G2, Cpu>, Case<G2, Gpu>>;
TYPED_TEST_SUITE(MsmTest, MsmCases);

TYPED_TEST(MsmTest, EachColumnIsTheSumOverSharedBases){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    for (const auto& shape : std::vector<Shape>{{1, 1}, {37, 1}, {300, 3}, {20, 50}, {0, 3}, {4, 0}}){
        const auto logs = scalars_with_zeros(shape.rows);
        const auto f = scalars_with_zeros(shape.rows * shape.cols);
        const auto bases = mul_generator<P>(Vec<Zp, E>::upload(logs));

        EXPECT_EQ(msm(bases, Vec<Zp, E>::upload(f), shape).download(), exponent_columns<P>(logs, f, shape))
            << shape.rows << "x" << shape.cols;
    }
}

TYPED_TEST(MsmTest, RejectsMismatchedShapes){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    const std::vector<ShapeMismatch> cases{
        {"more bases than rows", {2, 3}, 3, 6},
        {"fewer scalars than entries", {2, 3}, 2, 5},
        {"more scalars than entries", {2, 3}, 2, 7},
        {"scalars without bases", {0, 3}, 0, 1},
    };
    for (const auto& [name, shape, bases, scalars] : cases){
        const auto points = Vec<P, E>::upload(std::vector<P>(bases));
        const auto entries = Vec<Zp, E>::upload(std::vector<Zp>(scalars));
        EXPECT_THROW((void)msm(points, entries, shape), ShapeError) << name;
    }

    const auto points = Vec<P, E>::upload(std::vector<P>(2, P::generator()));
    const auto entries = Vec<Zp, E>::upload(std::vector<Zp>(6, Zp(1)));
    EXPECT_EQ(msm(points, entries, Shape{2, 3}).download(), std::vector<P>(3, P::generator() + P::generator()));
}
