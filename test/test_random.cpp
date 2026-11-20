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

    EXPECT_EQ(first.front().to_string(),
        "49265352440565724334159615326942584270622586368201043137340289674815210787403");
    EXPECT_EQ(first.back().to_string(),
        "30295430901326429246751547346519921365681185938484466553222178549211655845042");
    EXPECT_EQ(first, again);
    EXPECT_NE(first, other);
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
