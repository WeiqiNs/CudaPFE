#ifndef CUFE_VEC_STORAGE_HPP
#define CUFE_VEC_STORAGE_HPP

#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <cufe/core.hpp>
#include <cufe/engine.hpp>
#include <cufe/vec.hpp>
#include "curve/curve.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/access.hpp"
#include "support/device_array.hpp"
#include "support/runtime.hpp"

namespace cufe::detail{
    template <class T>
    struct DeviceElement;

    template <>
    struct DeviceElement<Zp>{ using type = Fr; };

    template <Side S>
    struct DeviceElement<Point<S>>{ using type = Affine<SideField<S>>; };

    template <>
    struct DeviceElement<Gt>{ using type = Fp12; };

    template <class T>
    using ElementOf = typename DeviceElement<T>::type;

    template <class T, Engine E>
    using Buffer = std::conditional_t<std::same_as<E, Cpu>, std::vector<T>, DeviceArray<T>>;

    template <class T, Engine E>
    struct Storage{
        Buffer<ElementOf<T>, E> buffer;
    };

    template <class T>
    [[nodiscard]] std::vector<T> to_host(const std::vector<T>& buffer){ return buffer; }

    template <class T>
    [[nodiscard]] std::vector<T> to_host(const DeviceArray<T>& buffer){ return buffer.copy_to_host(); }

    template <class T>
    [[nodiscard]] T element_at(const std::vector<T>& buffer, const std::size_t index){ return buffer[index]; }

    template <class T>
    [[nodiscard]] T element_at(const DeviceArray<T>& buffer, const std::size_t index){ return buffer.element(index); }

    template <Engine E, class T>
    [[nodiscard]] Buffer<T, E> to_engine(std::vector<T> values){
        if constexpr (std::same_as<E, Cpu>){
            return values;
        } else {
            DeviceArray<T> buffer(values.size());
            buffer.copy_from(values);
            return buffer;
        }
    }

    template <Engine E>
    void finish(){
        if constexpr (std::same_as<E, Gpu>) GpuRuntime::require().synchronize();
    }

    template <class T, Engine E>
    [[nodiscard]] const Buffer<ElementOf<T>, E>& buffer(const Vec<T, E>& values){
        return Access::storage(values).buffer;
    }

    template <class T, Engine E>
    [[nodiscard]] Vec<T, E> vec(Buffer<ElementOf<T>, E> buffer){
        return Access::wrap<Vec<T, E>>(std::make_shared<const Storage<T, E>>(Storage<T, E>{std::move(buffer)}));
    }

    template <class T>
    [[nodiscard]] std::vector<ElementOf<T>> to_elements(const std::span<const T> values){
        std::vector<ElementOf<T>> elements;
        elements.reserve(values.size());
        if constexpr (GroupPoint<T>){
            using F = decltype(ElementOf<T>::x);
            std::vector<Jacobian<F>> jacobians;
            jacobians.reserve(values.size());
            for (const auto& value : values) jacobians.push_back(Access::element<Jacobian<F>>(value));
            elements.resize(values.size());
            to_affine<F>(jacobians, elements);
        } else {
            for (const auto& value : values) elements.push_back(Access::element<ElementOf<T>>(value));
        }
        return elements;
    }

    template <class T>
    [[nodiscard]] T from_element(const ElementOf<T>& element){
        if constexpr (GroupPoint<T>) return Access::value<T>(from_affine(element));
        else return Access::value<T>(element);
    }

    template <class T, class Element>
    [[nodiscard]] std::vector<T> from_elements(const std::vector<Element>& elements){
        std::vector<T> values;
        values.reserve(elements.size());
        for (const auto& element : elements) values.push_back(from_element<T>(element));
        return values;
    }

    template <class X, class Y>
    void require_same_size(const X& x, const Y& y, const char* operation){
        if (x.size() != y.size()){
            throw ShapeError(std::string(operation) + " needs vectors of equal length, got "
                + std::to_string(x.size()) + " and " + std::to_string(y.size()));
        }
    }
}

#endif
