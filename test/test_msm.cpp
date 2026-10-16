#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>
#include <support/engines.hpp>

using namespace cudapfe;

namespace{
    struct ShapeMismatch{
        std::string_view name;
        MsmShape shape;
        std::size_t bases;
        std::size_t scalars;
    };

    struct ScalarMaker{
        std::string_view name;
        std::vector<Zp> (*make)(std::size_t);
    };

    std::vector<Zp> scalars_with_zeros(const std::size_t count){
        std::vector<Zp> scalars;
        for (std::size_t i = 0; i < count; ++i){
            scalars.push_back(i % 7 == 2 ? Zp() : i % 7 == 5 ? Zp(-1) : Zp::random());
        }
        return scalars;
    }

    std::vector<Zp> small_scalars(const std::size_t count){
        std::vector<Zp> scalars;
        for (std::size_t i = 0; i < count; ++i) scalars.push_back(Zp(static_cast<std::int64_t>(i % 9) - 4));
        return scalars;
    }

    std::vector<Zp> scalars_of_width(const std::size_t count, const std::size_t width){
        std::vector<Zp> scalars;
        for (std::size_t i = 0; i < count; ++i){
            if (width == 0 || i % 5 == 0){
                scalars.emplace_back();
                continue;
            }
            const auto low = std::uint64_t{1} << (width - 1);
            const auto magnitude = Zp(2).pow(width - 1) + Zp(i % low);
            scalars.push_back(i % 2 == 0 ? magnitude : -magnitude);
        }
        return scalars;
    }

    constexpr std::array<ScalarMaker, 2> kScalarMakers{{{"random", scalars_with_zeros}, {"small", small_scalars}}};

    std::vector<Zp> slice(const std::vector<Zp>& values, const Spread spread, const std::size_t segment,
        const std::size_t length){
        const auto offset = spread == Spread::shared ? 0 : segment * length;
        const auto first = values.begin() + static_cast<std::ptrdiff_t>(offset);
        return {first, first + static_cast<std::ptrdiff_t>(length)};
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
    for (const auto& [name, make_scalars] : kScalarMakers){
        for (const auto& shape : std::vector<Shape>{{1, 1}, {37, 1}, {300, 3}, {20, 50}, {0, 3}, {4, 0}}){
            const auto logs = scalars_with_zeros(shape.rows);
            const auto f = make_scalars(shape.rows * shape.cols);
            const auto bases = mul_generator<P>(Vec<Zp, E>::upload(logs));

            EXPECT_EQ(msm(bases, Vec<Zp, E>::upload(f), {1, shape}).download(), exponent_columns<P>(logs, f, shape))
                << name << " " << shape.rows << "x" << shape.cols;
        }
    }
}

TYPED_TEST(MsmTest, SegmentsTakeTheirOwnOrTheSharedBasesAndScalars){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    constexpr std::size_t segments = 3;
    constexpr Shape shape{10, 9};
    constexpr auto entries = shape.rows * shape.cols;
    for (const auto& [name, make_scalars] : kScalarMakers){
        for (const auto bases_spread : {Spread::per_segment, Spread::shared}){
            for (const auto scalars_spread : {Spread::per_segment, Spread::shared}){
                const auto logs = scalars_with_zeros(
                    bases_spread == Spread::shared ? shape.rows : segments * shape.rows);
                const auto f = make_scalars(scalars_spread == Spread::shared ? entries : segments * entries);
                std::vector<P> expected;
                for (std::size_t s = 0; s < segments; ++s){
                    const auto columns = exponent_columns<P>(
                        slice(logs, bases_spread, s, shape.rows), slice(f, scalars_spread, s, entries), shape);
                    expected.insert(expected.end(), columns.begin(), columns.end());
                }
                const auto bases = mul_generator<P>(Vec<Zp, E>::upload(logs));
                const MsmShape msm_shape{segments, shape, bases_spread, scalars_spread};

                EXPECT_EQ(msm(bases, Vec<Zp, E>::upload(f), msm_shape).download(), expected)
                    << name << " " << static_cast<int>(bases_spread) << static_cast<int>(scalars_spread);
            }
        }
    }
}

TYPED_TEST(MsmTest, LongColumnsOfSmallScalarsSumCorrectly){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    constexpr Shape shape{8192, 32};
    const auto logs = scalars_with_zeros(shape.rows);
    const auto f = small_scalars(shape.rows * shape.cols);
    const auto bases = mul_generator<P>(Vec<Zp, E>::upload(logs));

    EXPECT_EQ(msm(bases, Vec<Zp, E>::upload(f), {1, shape}).download(), exponent_columns<P>(logs, f, shape));
}

TYPED_TEST(MsmTest, ScalarsOfZeroAndBoundaryWidthsSumCorrectly){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    constexpr Shape shape{16, 16};
    const auto logs = scalars_with_zeros(shape.rows);
    const auto bases = mul_generator<P>(Vec<Zp, E>::upload(logs));
    for (const std::size_t width : {0, 32, 33}){
        const auto f = scalars_of_width(shape.rows * shape.cols, width);

        EXPECT_EQ(msm(bases, Vec<Zp, E>::upload(f), {1, shape}).download(), exponent_columns<P>(logs, f, shape))
            << width;
    }
}

TYPED_TEST(MsmTest, RejectsMismatchedShapes){
    using P = typename TypeParam::Value;
    using E = typename TypeParam::Engine;
    const std::vector<ShapeMismatch> cases{
        {"more bases than rows", {1, {2, 3}}, 3, 6},
        {"fewer scalars than entries", {1, {2, 3}}, 2, 5},
        {"more scalars than entries", {1, {2, 3}}, 2, 7},
        {"scalars without bases", {1, {0, 3}}, 0, 1},
        {"one segment of bases for two", {2, {2, 3}}, 2, 12},
        {"shared scalars for every segment", {2, {2, 3}, Spread::per_segment, Spread::shared}, 4, 12},
    };
    for (const auto& [name, shape, bases, scalars] : cases){
        const auto points = Vec<P, E>::upload(std::vector<P>(bases));
        const auto entries = Vec<Zp, E>::upload(std::vector<Zp>(scalars));
        EXPECT_THROW((void)msm(points, entries, shape), ShapeError) << name;
    }

    const auto points = Vec<P, E>::upload(std::vector<P>(2, P::generator()));
    const auto entries = Vec<Zp, E>::upload(std::vector<Zp>(6, Zp(1)));
    EXPECT_EQ(msm(points, entries, {1, {2, 3}}).download(), std::vector<P>(3, P::generator() + P::generator()));
}
