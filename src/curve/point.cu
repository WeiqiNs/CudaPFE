#include <cufe/core.hpp>
#include "curve/curve.hpp"
#include "curve/encoding.hpp"
#include "curve/generator_table.hpp"
#include "field/field.hpp"
#include "field/tower.hpp"
#include "support/access.hpp"

namespace cufe{
    using detail::Access;

    namespace{
        template <Side S>
        using Jacobian = detail::Jacobian<detail::SideField<S>>;

        template <Side S>
        Jacobian<S> jacobian(const Point<S>& p){ return Access::element<Jacobian<S>>(p); }

        template <Side S>
        Point<S> point(const Jacobian<S>& p){ return Access::value<Point<S>>(p); }

        template <Side S>
        constexpr bool sizes_match_encoding =
            Point<S>::compressed_size == detail::encoded_size<detail::SideField<S>>(Encoding::compressed)
            && Point<S>::uncompressed_size == detail::encoded_size<detail::SideField<S>>(Encoding::uncompressed);

        static_assert(sizes_match_encoding<Side::g1> && sizes_match_encoding<Side::g2>);
    }

    template <Side S>
    Point<S> Point<S>::generator(){
        using F = detail::SideField<S>;
        return point<S>(detail::from_affine(detail::CurveParams<F>::generator()));
    }

    template <Side S>
    Point<S> Point<S>::random(){
        return mul_generator(Zp::random());
    }

    template <Side S>
    Point<S> Point<S>::mul_generator(const Zp& scalar){
        const auto& table = detail::generator_table<detail::SideField<S>>();
        return point<S>(detail::fixed_base_mul(table.data(), Access::element<detail::Fr>(scalar)));
    }

    template <Side S>
    Point<S> Point<S>::from_bytes(const ByteView bytes){
        return point<S>(detail::from_affine(detail::decode<detail::SideField<S>>(bytes)));
    }

    template <Side S>
    Bytes Point<S>::to_bytes(const Encoding encoding) const{
        return detail::encode(detail::to_affine(jacobian(*this)), encoding);
    }

    template <Side S>
    bool Point<S>::is_identity() const{
        return jacobian(*this).is_identity();
    }

    template <Side S>
    Point<S> Point<S>::plus(const Point& q) const{
        return point<S>(detail::add(jacobian(*this), jacobian(q)));
    }

    template <Side S>
    Point<S> Point<S>::times(const Zp& k) const{
        return point<S>(detail::mul(jacobian(*this), Access::element<detail::Fr>(k)));
    }

    template <Side S>
    Point<S> Point<S>::negated() const{
        return point<S>(detail::neg(jacobian(*this)));
    }

    template <Side S>
    bool Point<S>::equals(const Point& q) const{
        return detail::equal(jacobian(*this), jacobian(q));
    }

    template class Point<Side::g1>;
    template class Point<Side::g2>;
}
