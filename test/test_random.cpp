#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>

using namespace cudapfe;

TEST(RandomTest, SeedMakesTheRandomSequenceReproducible){
    seed(bytes_of("seed-a"));
    const std::vector first{Zp::random(), Zp::random()};
    seed(bytes_of("seed-a"));
    const std::vector again{Zp::random(), Zp::random()};
    seed(bytes_of("seed-b"));
    const std::vector other{Zp::random(), Zp::random()};

    EXPECT_EQ(first, again);
    EXPECT_NE(first, other);
    EXPECT_NE(first.front(), first.back());
    EXPECT_THROW(seed(Bytes{}), ShapeError);
}
