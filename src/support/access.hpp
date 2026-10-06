#ifndef CUDAPFE_SUPPORT_ACCESS_HPP
#define CUDAPFE_SUPPORT_ACCESS_HPP

#include <bit>
#include <memory>
#include <utility>
#include <cudapfe/core.hpp>

namespace cudapfe::detail{
    struct Access{
        template <class Element, class T>
        [[nodiscard]] static Element element(const T& x){ return std::bit_cast<Element>(x.words_); }

        template <class T, class Element>
        [[nodiscard]] static T value(const Element& element){
            T x;
            x.words_ = std::bit_cast<decltype(x.words_)>(element);
            return x;
        }

        template <class Handle>
        [[nodiscard]] static const auto& storage(const Handle& handle){ return *handle.storage_; }

        template <class Handle, class Storage>
        [[nodiscard]] static Handle wrap(std::shared_ptr<const Storage> storage){ return Handle(std::move(storage)); }
    };
}

#endif
