#ifndef CUFE_FIELD_CONSTANTS_HPP
#define CUFE_FIELD_CONSTANTS_HPP

#include "support/words.hpp"

namespace cufe::detail{
    inline constexpr Words<6> kP{
        0xb9feffffffffaaab, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
        0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    };
    inline constexpr Words<4> kR{0xffffffff00000001, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48};

    inline constexpr Words<6> kFpInverseExponent = sub_word(kP, 2);
    inline constexpr Words<6> kFpSqrtExponent = shift_right(add_word(kP, 1), 2);
    inline constexpr Words<6> kFp2SqrtExponent = shift_right(sub_word(kP, 3), 2);
    inline constexpr Words<6> kFpHalfExponent = shift_right(sub_word(kP, 1), 1);
    inline constexpr Words<4> kFrInverseExponent = sub_word(kR, 2);
}

#endif
