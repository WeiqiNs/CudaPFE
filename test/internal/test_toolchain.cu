#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <ff/bls12-381.hpp>
#include <support/engines.hpp>
#include <support/oracle.hpp>
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"

using namespace cufe;
using namespace cufe::detail;

namespace{
    constexpr int kSquarings = 1000;
    constexpr Word kGolden = 0x9e3779b97f4a7c15;

    static_assert(sizeof(bls12_381::fp_t) == 48 && sizeof(bls12_381::fr_t) == 32);

    template <class F>
    using LimbsOf = std::array<std::uint32_t, sizeof(F) / sizeof(std::uint32_t)>;

    template <class F>
    using WordsOf = Words<sizeof(F) / sizeof(Word)>;

    template <class F>
    struct SquareRepeatedly{
        const std::uint32_t* input;
        std::uint32_t* output;

        CUFE_HD void operator()(std::size_t) const{
#ifdef __CUDA_ARCH__
            F x;
            for (std::size_t i = 0; i < x.len(); ++i) x[i] = input[i];
            for (int i = 0; i < kSquarings; ++i) x = sqr(x);
            x.store(output);
#endif
        }
    };

    struct AddIndexHash{
        Word* slots;

        CUFE_HD void operator()(const std::size_t index) const{ slots[index] += index * kGolden + 1; }
    };

    struct AddIndexHashInWarps : AddIndexHash{
        static constexpr int threads = 32;
    };

    template <class F>
    WordsOf<F> words_of(const F& x){ return std::bit_cast<WordsOf<F>>(x); }

    template <class F>
    F squared_repeatedly(F x){
        for (int i = 0; i < kSquarings; ++i) x = sqr(x);
        return x;
    }

    template <class F>
    WordsOf<F> squared_repeatedly_on_device(const WordsOf<F>& input){
        DeviceArray<std::uint32_t> limbs(sizeof(F) / sizeof(std::uint32_t));
        DeviceArray<std::uint32_t> result(limbs.size());
        limbs.copy_from(std::bit_cast<LimbsOf<F>>(input));
        for_each<Gpu>(1, SquareRepeatedly<F>{limbs.data(), result.data()});
        const auto host = result.copy_to_host();
        LimbsOf<F> output;
        std::copy(host.begin(), host.end(), output.begin());
        return std::bit_cast<WordsOf<F>>(output);
    }

    template <Engine E, class Op>
    std::vector<Word> visit(const std::size_t count){
        const std::vector<Word> zeros(count + 1, 0);
        if constexpr (std::same_as<E, Gpu>){
            DeviceArray<Word> slots(zeros.size());
            slots.copy_from(zeros);
            for_each<Gpu>(count, Op{slots.data()});
            return slots.copy_to_host();
        } else {
            auto slots = zeros;
            for_each<Cpu>(count, Op{slots.data()});
            return slots;
        }
    }
}

TEST(ToolchainTest, HostMontgomeryArithmeticMatchesBlst){
    blst_fp fp;
    blst_fr fr;
    const std::array<std::uint64_t, 6> three{3};
    blst_fp_from_uint64(&fp, three.data());
    blst_fr_from_uint64(&fr, three.data());
    for (int i = 0; i < kSquarings; ++i){
        blst_fp_sqr(&fp, &fp);
        blst_fr_sqr(&fr, &fr);
    }

    EXPECT_EQ(words_of(squared_repeatedly(bls12_381::fp_t(3))), std::bit_cast<Words<6>>(fp));
    EXPECT_EQ(words_of(squared_repeatedly(bls12_381::fr_t(3))), std::bit_cast<Words<4>>(fr));
}

TEST(ToolchainTest, DeviceMontgomeryArithmeticMatchesHost){
    CUFE_REQUIRE_GPU();
    const bls12_381::fp_t fp(3);
    const bls12_381::fr_t fr(3);

    EXPECT_EQ(squared_repeatedly_on_device<bls12_381::fp_t>(words_of(fp)), words_of(squared_repeatedly(fp)));
    EXPECT_EQ(squared_repeatedly_on_device<bls12_381::fr_t>(words_of(fr)), words_of(squared_repeatedly(fr)));
}

template <class E>
class ForEachTest : public EngineTest<E>{};

TYPED_TEST_SUITE(ForEachTest, Engines);

TYPED_TEST(ForEachTest, VisitsEveryIndexExactlyOnce){
    for (const std::size_t count : std::array<std::size_t, 6>{0, 1, 127, 128, 129, 100003}){
        std::vector<Word> expected(count + 1, 0);
        for (std::size_t index = 0; index < count; ++index) expected[index] = index * kGolden + 1;

        EXPECT_EQ((visit<TypeParam, AddIndexHash>(count)), expected) << count << " indices";
        EXPECT_EQ((visit<TypeParam, AddIndexHashInWarps>(count)), expected) << count << " indices in warps";
    }
}

TEST(DeviceArrayTest, AllocationFailureIsADeviceError){
    CUFE_REQUIRE_GPU();
    EXPECT_THROW(DeviceArray<std::byte>(std::size_t{1} << 50), DeviceError);
    EXPECT_EQ((visit<Gpu, AddIndexHash>(1)), (std::vector<Word>{1, 0}));
}
