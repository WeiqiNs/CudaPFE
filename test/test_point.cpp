#include <concepts>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <cudapfe/cudapfe.hpp>

using namespace cudapfe;

namespace{
    class PointNames{
    public:
        template <class P>
        static std::string GetName(int){ return std::same_as<P, G1> ? "G1" : "G2"; }
    };
}

template <class P>
class PointTest : public ::testing::Test{};

using Points = ::testing::Types<G1, G2>;
TYPED_TEST_SUITE(PointTest, Points, PointNames);

TYPED_TEST(PointTest, GroupLawsHold){
    using P = TypeParam;
    const auto p = P::random(), q = P::random();

    EXPECT_EQ(p + P(), p);
    EXPECT_TRUE((p - p).is_identity());
    EXPECT_TRUE((-p + p).is_identity());
    EXPECT_EQ(p + q, q + p);
    EXPECT_EQ(p * Zp(2), p + p);
    EXPECT_EQ(Zp(3) * p, p + p + p);
    EXPECT_FALSE(p.is_identity());

    auto acc = p;
    acc += q;
    acc -= p;
    acc *= 2;
    EXPECT_EQ(acc, q + q);
}

TYPED_TEST(PointTest, GeneratorHasTheGroupOrder){
    using P = TypeParam;
    const auto x = Zp::random();

    EXPECT_EQ(P::mul_generator(x), P::generator() * x);
    EXPECT_TRUE((P::generator() * Zp(-1) + P::generator()).is_identity());
    EXPECT_TRUE(P::mul_generator(Zp()).is_identity());

    const auto y = Zp::random();
    EXPECT_EQ(P::mul_generator(Vector{x, Zp(), y}), (std::vector{P::generator() * x, P(), P::generator() * y}));
    EXPECT_TRUE(P::mul_generator(Vector{}).empty());
    EXPECT_TRUE(P::mul_generator({}).is_identity());
    EXPECT_EQ(P::mul_generator({x}), P::generator() * x);
}

TYPED_TEST(PointTest, EveryEncodingRoundTrips){
    using P = TypeParam;
    const auto p = P::random();

    EXPECT_EQ(P::from_bytes(p.to_bytes()), p);
    EXPECT_EQ(P::from_bytes(p.to_bytes(Encoding::uncompressed)), p);
    EXPECT_EQ(p.to_bytes().size(), P::compressed_size);
    EXPECT_EQ(p.to_bytes(Encoding::uncompressed).size(), P::uncompressed_size);

    Bytes identity(P::compressed_size, 0);
    identity.front() = 0xc0;
    EXPECT_EQ(P().to_bytes(), identity);
    EXPECT_TRUE(P::from_bytes(P().to_bytes()).is_identity());
    EXPECT_TRUE(P::from_bytes(P().to_bytes(Encoding::uncompressed)).is_identity());
}
