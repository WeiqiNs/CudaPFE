#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include <gtest/gtest.h>
#include "scheme_types.hpp"

using cufe::QFE::IntMatrix;
using cufe::QFE::IntVec;

namespace{
    constexpr std::int64_t kEntryBound = 2;

    std::int64_t quadratic_form(const IntVec& x, const IntMatrix& f, const IntVec& y){
        std::int64_t total = 0;
        for (std::size_t i = 0; i < x.size(); ++i){
            for (std::size_t j = 0; j < y.size(); ++j) total += x[i] * f[i][j] * y[j];
        }
        return total;
    }

    std::vector<IntMatrix> random_matrices(const std::size_t count, const std::size_t size){
        std::vector<IntMatrix> matrices;
        for (std::size_t i = 0; i < count; ++i) matrices.push_back(random_rows(size, size, kEntryBound));
        return matrices;
    }
}

template <class Scheme>
class QuadraticTest : public CaseTest<Scheme>{};

TYPED_TEST_SUITE(QuadraticTest, QuadraticSchemes);

TYPED_TEST(QuadraticTest, DecryptsQuadraticFormsExactlyWithinTheRange){
    const auto keys = TypeParam::setup(3);
    const auto decrypt = TypeParam::decryptor(keys, kRange);
    const auto sk = keygen(keys.msk, IntMatrix{{1, 0, 2}, {0, -1, 0}, {3, 1, 1}});

    EXPECT_EQ(decrypt(sk, enc(keys.pk, IntVec{1, -2, 3}, IntVec{4, 5, -6})), Results{35});
    EXPECT_EQ(decrypt(sk, enc(keys.pk, IntVec{-1, 1, 0}, IntVec{2, 0, 1})), Results{-4});
    EXPECT_EQ(decrypt(sk, enc(keys.pk, IntVec{0, 0, 0}, IntVec{4, 5, -6})), Results{0});
    EXPECT_EQ(decrypt(sk, enc(keys.pk, IntVec{1, 0, 0}, IntVec{100, 0, 0})), Results{100});
    EXPECT_EQ(decrypt(sk, enc(keys.pk, IntVec{0, 1, 0}, IntVec{0, 100, 0})), Results{-100});
    EXPECT_EQ(decrypt(sk, enc(keys.pk, IntVec{1, 0, 0}, IntVec{101, 0, 0})), Results{std::nullopt});
    EXPECT_EQ(decrypt(sk, enc(keys.pk, IntVec{0, 1, 0}, IntVec{0, 101, 0})), Results{std::nullopt});
}

TYPED_TEST(QuadraticTest, DecryptsOneCiphertextUnderManyKeys){
    const auto keys = TypeParam::setup(3);
    const auto decrypt = TypeParam::decryptor(keys, kRange);
    const auto ct = enc(keys.pk, IntVec{1, -2, 3}, IntVec{4, 5, -6});
    const std::vector<IntMatrix> functions{
        {{1, 0, 2}, {0, -1, 0}, {3, 1, 1}},
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 1}},
        {{30, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    };

    EXPECT_EQ(decrypt(keygen(keys.msk, functions), ct), (Results{35, -24, -18, std::nullopt}));
}

TYPED_TEST(QuadraticTest, DecryptsManyCiphertextsUnderOneKey){
    const auto keys = TypeParam::setup(3);
    const auto decrypt = TypeParam::decryptor(keys, kRange);
    const IntMatrix function{{2, -1, 0}, {0, 1, 1}, {-2, 0, 1}};
    const auto lefts = random_rows(8, 3, kEntryBound);
    const auto rights = random_rows(8, 3, kEntryBound);
    Results expected;
    for (std::size_t i = 0; i < lefts.size(); ++i) expected.push_back(quadratic_form(lefts[i], function, rights[i]));

    EXPECT_EQ(decrypt(keygen(keys.msk, function), enc(keys.pk, lefts, rights)), expected);
}

TYPED_TEST(QuadraticTest, DecryptsABatchPairwise){
    const auto keys = TypeParam::setup(3);
    const auto decrypt = TypeParam::decryptor(keys, kRange);
    const auto functions = random_matrices(8, 3);
    const auto lefts = random_rows(8, 3, kEntryBound);
    const auto rights = random_rows(8, 3, kEntryBound);
    Results expected;
    for (std::size_t i = 0; i < lefts.size(); ++i){
        expected.push_back(quadratic_form(lefts[i], functions[i], rights[i]));
    }

    EXPECT_EQ(decrypt(keygen(keys.msk, functions), enc(keys.pk, lefts, rights)), expected);
}

TYPED_TEST(QuadraticTest, HandlesVectorsOfLengthOne){
    const auto keys = TypeParam::setup(1);
    const auto decrypt = TypeParam::decryptor(keys, kRange);

    EXPECT_EQ(decrypt(keygen(keys.msk, IntMatrix{{-3}}), enc(keys.pk, IntVec{7}, IntVec{2})), Results{-42});
}

TYPED_TEST(QuadraticTest, RejectsInputsOfTheWrongShape){
    const auto keys = TypeParam::setup(3);

    EXPECT_THROW((void)keygen(keys.msk, IntMatrix{{1, 2}, {3, 4}, {5, 6}}), cufe::ShapeError);
    EXPECT_THROW((void)keygen(keys.msk, IntMatrix{{1, 2}, {3, 4}}), cufe::ShapeError);
    EXPECT_THROW((void)enc(keys.pk, IntVec{1, 2}, IntVec{1, 2, 3}), cufe::ShapeError);
    EXPECT_THROW((void)enc(keys.pk, IntVec{1, 2, 3}, IntVec{1, 2, 3, 4}), cufe::ShapeError);
    EXPECT_THROW((void)enc(keys.pk, IntMatrix{{1, 2, 3}, {4, 5, 6}}, IntMatrix{{1, 2, 3}}), cufe::ShapeError);
}

TYPED_TEST(QuadraticTest, RejectsIncompatibleBatchCounts){
    const auto keys = TypeParam::setup(2);
    const auto decrypt = TypeParam::decryptor(keys, kRange);
    const auto sks = keygen(keys.msk, std::vector<IntMatrix>{{{1, 0}, {0, 1}}, {{0, 1}, {0, 0}}});
    const IntMatrix lefts{{1, 2}, {3, 4}, {5, 6}};
    const IntMatrix rights{{1, 1}, {0, 1}, {1, 0}};

    EXPECT_THROW((void)decrypt(sks, enc(keys.pk, lefts, rights)), cufe::ShapeError);
    EXPECT_EQ(decrypt(sks, enc(keys.pk, IntMatrix{lefts[0], lefts[1]}, IntMatrix{rights[0], rights[1]})),
        (Results{3, 3}));
    EXPECT_TRUE(decrypt(keygen(keys.msk, std::vector<IntMatrix>{}), enc(keys.pk, IntMatrix{}, IntMatrix{})).empty());
}
