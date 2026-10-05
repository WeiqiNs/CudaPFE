#include <vector>
#include <gtest/gtest.h>
#include <cufe/cufe.hpp>

using namespace cufe;

namespace{
    Vector vector(const std::vector<int>& values){
        return Vector(values.begin(), values.end());
    }
}

TEST(VectorTest, VectorOperationsAreElementwise){
    const auto x = vector({1, 2, 3});
    const auto y = vector({4, 5, 6});

    EXPECT_EQ(x + y, vector({5, 7, 9}));
    EXPECT_EQ(x - y, vector({-3, -3, -3}));
    EXPECT_EQ(x * Zp(2), vector({2, 4, 6}));
    EXPECT_EQ(Zp(2) * x, vector({2, 4, 6}));
    EXPECT_EQ(hadamard(x, y), vector({4, 10, 18}));
    EXPECT_EQ(inner(x, y), Zp(32));
    EXPECT_EQ(sum(x), Zp(6));
    EXPECT_EQ(concat({x, y, vector({7})}), vector({1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(random_vector(4).size(), 4u);

    EXPECT_THROW((void)(x + vector({1, 2})), ShapeError);
}
