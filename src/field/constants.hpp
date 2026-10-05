#ifndef CUFE_FIELD_CONSTANTS_HPP
#define CUFE_FIELD_CONSTANTS_HPP

#include <array>
#include "support/words.hpp"

namespace cufe::detail{
    using Fp2Words = std::array<Words<6>, 2>;

    inline constexpr Words<6> kP{
        0xb9feffffffffaaab, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
        0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    };
    inline constexpr Words<4> kR{0xffffffff00000001, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48};

    inline constexpr Words<6> kFpInverseExponent = sub_word(kP, 2);
    inline constexpr Words<6> kFpSqrtExponent = shift_right(add_word(kP, 1), 2);
    inline constexpr Words<4> kFrInverseExponent = sub_word(kR, 2);

    inline constexpr Words<6> kFpMontgomeryOne{
        0x760900000002fffd, 0xebf4000bc40c0002, 0x5f48985753c758ba,
        0x77ce585370525745, 0x5c071a97a256ec6d, 0x15f65ec3fa80e493
    };

    inline constexpr std::array<Fp2Words, 3> kFrobeniusFp6C1{
        Fp2Words{
            Words<6>{},
            Words<6>{
                0xcd03c9e48671f071, 0x5dab22461fcda5d2, 0x587042afd3851b95,
                0x8eb60ebe01bacb9e, 0x03f97d6e83d050d2, 0x18f0206554638741
            }
        },
        Fp2Words{
            Words<6>{
                0x30f1361b798a64e8, 0xf3b8ddab7ece5a2a, 0x16a8ca3ac61577f7,
                0xc26a2ff874fd029b, 0x3636b76660701c6e, 0x051ba4ab241b6160
            },
            Words<6>{}
        },
        Fp2Words{
            Words<6>{},
            kFpMontgomeryOne
        },
    };

    inline constexpr std::array<Words<6>, 3> kFrobeniusFp6C2{
        Words<6>{
            0x890dc9e4867545c3, 0x2af322533285a5d5, 0x50880866309b7e2c,
            0xa20d1b8c7e881024, 0x14e4f04fe2db9068, 0x14e56d3f1564853a
        },
        Words<6>{
            0xcd03c9e48671f071, 0x5dab22461fcda5d2, 0x587042afd3851b95,
            0x8eb60ebe01bacb9e, 0x03f97d6e83d050d2, 0x18f0206554638741
        },
        Words<6>{
            0x43f5fffffffcaaae, 0x32b7fff2ed47fffd, 0x07e83a49a2e99d69,
            0xeca8f3318332bb7a, 0xef148d1ea0f4c069, 0x040ab3263eff0206
        },
    };

    inline constexpr std::array<Fp2Words, 3> kFrobeniusFp12{
        Fp2Words{
            Words<6>{
                0x07089552b319d465, 0xc6695f92b50a8313, 0x97e83cccd117228f,
                0xa35baecab2dc29ee, 0x1ce393ea5daace4d, 0x08f2220fb0fb66eb
            },
            Words<6>{
                0xb2f66aad4ce5d646, 0x5842a06bfc497cec, 0xcf4895d42599d394,
                0xc11b9cba40a8e8d0, 0x2e3813cbe5a0de89, 0x110eefda88847faf
            }
        },
        Fp2Words{
            Words<6>{
                0xecfb361b798dba3a, 0xc100ddb891865a2c, 0x0ec08ff1232bda8e,
                0xd5c13cc6f1ca4721, 0x47222a47bf7b5c04, 0x0110f184e51c5f59
            },
            Words<6>{}
        },
        Fp2Words{
            Words<6>{
                0x3e2f585da55c9ad1, 0x4294213d86c18183, 0x382844c88b623732,
                0x92ad2afd19103e18, 0x1d794e4fac7cf0b9, 0x0bd592fc7d825ec8
            },
            Words<6>{
                0x7bcfa7a25aa30fda, 0xdc17dec12a927e7c, 0x2f088dd86b4ebef1,
                0xd1ca2087da74d4a7, 0x2da2596696cebc1d, 0x0e2b7eedbbfd87d2
            }
        },
    };
}

#endif
