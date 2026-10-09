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

TEST(RandomTest, RandomVectorDrawsTheScalarSequence){
    seed(bytes_of("stream"));
    const auto drawn = random_vector(300);
    const auto next = Zp::random();
    seed(bytes_of("stream"));
    std::vector<Zp> sequence(301);
    for (auto& x : sequence) x = Zp::random();

    EXPECT_EQ(drawn, std::vector(sequence.begin(), sequence.end() - 1));
    EXPECT_EQ(next, sequence.back());
}
