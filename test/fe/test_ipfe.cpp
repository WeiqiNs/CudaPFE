#include <cstddef>
#include <cstdint>
#include <optional>
#include <gtest/gtest.h>
#include "scheme_types.hpp"

using cufe::IPFE::IntMatrix;
using cufe::IPFE::IntVec;

namespace{
    constexpr std::int64_t kEntryBound = 5;

    std::int64_t inner_product(const IntVec& x, const IntVec& y){
        std::int64_t total = 0;
        for (std::size_t i = 0; i < x.size(); ++i) total += x[i] * y[i];
        return total;
    }

    Results pairwise_products(const IntMatrix& functions, const IntMatrix& messages){
        Results products;
        for (std::size_t i = 0; i < functions.size(); ++i) products.push_back(inner_product(functions[i], messages[i]));
        return products;
    }
}

template <class Scheme>
class InnerProductTest : public CaseTest<Scheme>{};

TYPED_TEST_SUITE(InnerProductTest, InnerProductSchemes);

TYPED_TEST(InnerProductTest, DecryptsInnerProductsExactlyWithinTheRange){
    const auto msk = TypeParam::setup(4);
    const auto decrypt = TypeParam::decryptor(msk, kRange);
    const auto sk = keygen(msk, IntVec{1, -2, 3, 4});

    EXPECT_EQ(decrypt(sk, enc(msk, IntVec{5, 6, 7, 8})), Results{46});
    EXPECT_EQ(decrypt(sk, enc(msk, IntVec{-4, 5, -6, 0})), Results{-32});
    EXPECT_EQ(decrypt(sk, enc(msk, IntVec{2, 1, 0, 0})), Results{0});
    EXPECT_EQ(decrypt(sk, enc(msk, IntVec{0, 0, 0, 25})), Results{100});
    EXPECT_EQ(decrypt(sk, enc(msk, IntVec{0, 0, 0, -25})), Results{-100});
    EXPECT_EQ(decrypt(sk, enc(msk, IntVec{1, 0, 0, 25})), Results{std::nullopt});
    EXPECT_EQ(decrypt(sk, enc(msk, IntVec{-1, 0, 0, -25})), Results{std::nullopt});
}

TYPED_TEST(InnerProductTest, DecryptsOneCiphertextUnderManyKeys){
    const auto msk = TypeParam::setup(4);
    const auto decrypt = TypeParam::decryptor(msk, kRange);
    const auto ct = enc(msk, IntVec{5, 6, 7, 8});
    const auto keys = keygen(msk, IntMatrix{{1, -2, 3, 4}, {0, 0, 0, 1}, {1, 1, 1, 1}, {20, 0, 0, 1}});

    EXPECT_EQ(decrypt(keys, ct), (Results{46, 8, 26, std::nullopt}));
}

TYPED_TEST(InnerProductTest, DecryptsManyCiphertextsUnderOneKey){
    const auto msk = TypeParam::setup(4);
    const auto decrypt = TypeParam::decryptor(msk, kRange);
    const IntVec function{3, -1, 0, 2};
    const auto messages = random_rows(12, 4, kEntryBound);
    Results expected;
    for (const auto& message : messages) expected.push_back(inner_product(function, message));

    EXPECT_EQ(decrypt(keygen(msk, function), enc(msk, messages)), expected);
}

TYPED_TEST(InnerProductTest, DecryptsABatchPairwise){
    const auto msk = TypeParam::setup(4);
    const auto decrypt = TypeParam::decryptor(msk, kRange);
    const auto functions = random_rows(16, 4, kEntryBound);
    const auto messages = random_rows(16, 4, kEntryBound);

    EXPECT_EQ(decrypt(keygen(msk, functions), enc(msk, messages)), pairwise_products(functions, messages));
}

TYPED_TEST(InnerProductTest, PreparedKeyDecryptsEveryCiphertextLikeTheKey){
    const auto msk = TypeParam::setup(4);
    const auto decrypt = TypeParam::decryptor(msk, kRange);
    const auto sk = keygen(msk, IntVec{1, -2, 3, 4});
    const auto ct = enc(msk, IntMatrix{{5, 6, 7, 8}, {-4, 5, -6, 0}, {0, 0, 0, 25}, {1, 0, 0, 25}});
    const Results expected{46, -32, 100, std::nullopt};

    EXPECT_EQ(decrypt(prepare(sk), ct), expected);
    EXPECT_EQ(decrypt(sk, ct), expected);

    const IntMatrix functions{{1, 2, 3, 4}, {-1, 0, 2, 0}};
    const IntMatrix messages{{4, 3, 2, 1}, {7, 7, 7, 7}};
    EXPECT_EQ(decrypt(prepare(keygen(msk, functions)), enc(msk, messages)), (Results{20, 7}));
}

TYPED_TEST(InnerProductTest, HandlesVectorsOfLengthOne){
    const auto msk = TypeParam::setup(1);
    const auto decrypt = TypeParam::decryptor(msk, kRange);

    EXPECT_EQ(decrypt(keygen(msk, IntVec{-3}), enc(msk, IntVec{7})), Results{-21});
    EXPECT_EQ(decrypt(prepare(keygen(msk, IntVec{-3})), enc(msk, IntVec{7})), Results{-21});
}

TYPED_TEST(InnerProductTest, RejectsVectorsOfTheWrongLength){
    const auto msk = TypeParam::setup(4);

    EXPECT_THROW((void)keygen(msk, IntVec{1, 2, 3}), cufe::ShapeError);
    EXPECT_THROW((void)keygen(msk, IntVec{1, 2, 3, 4, 5}), cufe::ShapeError);
    EXPECT_THROW((void)keygen(msk, IntMatrix{{1, 2, 3, 4}, {1, 2, 3}}), cufe::ShapeError);
    EXPECT_THROW((void)enc(msk, IntVec{1, 2, 3}), cufe::ShapeError);
    EXPECT_THROW((void)enc(msk, IntVec{1, 2, 3, 4, 5}), cufe::ShapeError);
    EXPECT_THROW((void)enc(msk, IntMatrix{{1, 2, 3, 4}, {1, 2, 3, 4, 5}}), cufe::ShapeError);
}

TYPED_TEST(InnerProductTest, RejectsIncompatibleBatchCounts){
    const auto msk = TypeParam::setup(2);
    const auto decrypt = TypeParam::decryptor(msk, kRange);
    const auto keys = keygen(msk, IntMatrix{{1, 2}, {3, 4}});

    EXPECT_THROW((void)decrypt(keys, enc(msk, IntMatrix{{1, 0}, {0, 1}, {1, 1}})), cufe::ShapeError);
    EXPECT_THROW((void)decrypt(prepare(keys), enc(msk, IntMatrix{{1, 0}, {0, 1}, {1, 1}})), cufe::ShapeError);
    EXPECT_EQ(decrypt(keys, enc(msk, IntMatrix{{1, 0}, {0, 1}})), (Results{1, 4}));
    EXPECT_TRUE(decrypt(keygen(msk, IntMatrix{}), enc(msk, IntMatrix{})).empty());
}

TEST(MatrixSchemeSetupTest, RefusesMasterKeysThatDoNotFitOnTheDevice){
    CUFE_REQUIRE_GPU();
    constexpr std::size_t n = 100000;

    EXPECT_THROW((void)cufe::IPFE::BJK::setup<cufe::Gpu>(n), cufe::DeviceError);
    EXPECT_THROW((void)cufe::IPFE::TAO::setup<cufe::Gpu>(n), cufe::DeviceError);
    EXPECT_THROW((void)cufe::IPFE::KIM::setup<cufe::Gpu>(2 * n), cufe::DeviceError);
}
