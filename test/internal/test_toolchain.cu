#include <array>
#include <cstddef>
#include <vector>
#include <support/engines.hpp>
#include "support/device_array.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"

using namespace cudapfe;
using namespace cudapfe::detail;

namespace{
    constexpr Word kGolden = 0x9e3779b97f4a7c15;

    struct AddIndexHash{
        Word* slots;

        CUDAPFE_HD void operator()(const std::size_t index) const{ slots[index] += index * kGolden + 1; }
    };

    struct AddIndexHashInWarps : AddIndexHash{
        static constexpr int threads = 32;
    };

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
    CUDAPFE_REQUIRE_GPU();
    EXPECT_THROW(DeviceArray<std::byte>(std::size_t{1} << 50), DeviceError);
    EXPECT_EQ((visit<Gpu, AddIndexHash>(1)), (std::vector<Word>{1, 0}));
}
