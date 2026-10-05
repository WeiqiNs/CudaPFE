#include <cstddef>
#include <vector>
#include <blst.h>
#include <gtest/gtest.h>
#include <cufe/cufe.hpp>

using namespace cufe;

namespace{
    constexpr std::size_t kMultiPairs = 257;
    constexpr std::size_t kOraclePairs = 10;

    Bytes blst_gt_bytes(const blst_p1_affine& p, const blst_p2_affine& q){
        blst_fp12 miller, value;
        blst_miller_loop(&miller, &q, &p);
        blst_final_exp(&value, &miller);
        Bytes out(Gt::byte_size);
        blst_bendian_from_fp12(out.data(), &value);
        return out;
    }
}

TEST(PairingTest, IsBilinearAndNonDegenerate){
    const auto a = Zp::random(), b = Zp::random();
    const auto p = G1::random(), p2 = G1::random();
    const auto q = G2::random();

    EXPECT_FALSE(Gt::generator().is_one());
    EXPECT_EQ(pair(p * a, q * b), pair(p, q).pow(a * b));
    EXPECT_EQ(pair(p + p2, q), pair(p, q) * pair(p2, q));
    EXPECT_TRUE(pair(G1(), q).is_one());
    EXPECT_TRUE(pair(p, G2()).is_one());
}

TEST(PairingTest, MultiPairingIsTheProduct){
    std::vector<G1> ps;
    std::vector<G2> qs;
    Gt product;
    for (std::size_t i = 0; i < kMultiPairs; ++i){
        ps.push_back(G1::random());
        qs.push_back(G2::random());
        product *= pair(ps.back(), qs.back());
    }
    ps.insert(ps.begin() + 3, G1());
    qs.insert(qs.begin() + 3, G2::random());
    ps.insert(ps.begin() + 100, G1::random());
    qs.insert(qs.begin() + 100, G2());

    EXPECT_EQ(pair(ps, qs), product);
    EXPECT_TRUE(pair(std::vector<G1>{}, std::vector<G2>{}).is_one());
    qs.pop_back();
    EXPECT_THROW((void)pair(ps, qs), ShapeError);
}

TEST(PairingTest, GeneratorPairingMatchesBlst){
    blst_p1_affine g1;
    blst_p2_affine g2;
    blst_p1_to_affine(&g1, blst_p1_generator());
    blst_p2_to_affine(&g2, blst_p2_generator());
    EXPECT_EQ(Gt::generator().to_bytes(), blst_gt_bytes(g1, g2));
}

TEST(PairingTest, RandomPairingsMatchBlst){
    for (std::size_t i = 0; i < kOraclePairs; ++i){
        const auto p = G1::random();
        const auto q = G2::random();
        blst_p1_affine blst_p;
        blst_p2_affine blst_q;
        ASSERT_EQ(blst_p1_deserialize(&blst_p, p.to_bytes(Encoding::uncompressed).data()), BLST_SUCCESS);
        ASSERT_EQ(blst_p2_deserialize(&blst_q, q.to_bytes(Encoding::uncompressed).data()), BLST_SUCCESS);
        EXPECT_EQ(pair(p, q).to_bytes(), blst_gt_bytes(blst_p, blst_q)) << i;
    }
}
