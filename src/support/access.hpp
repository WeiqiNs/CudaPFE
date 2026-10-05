#ifndef CUFE_SUPPORT_ACCESS_HPP
#define CUFE_SUPPORT_ACCESS_HPP

#include <bit>
#include <cufe/core.hpp>

namespace cufe::detail{
    struct Access{
        template <class Element, class T>
        [[nodiscard]] static Element element(const T& x){ return std::bit_cast<Element>(x.words_); }

        template <class T, class Element>
        [[nodiscard]] static T value(const Element& element){
            T x;
            x.words_ = std::bit_cast<decltype(x.words_)>(element);
            return x;
        }
    };
}

#endif
