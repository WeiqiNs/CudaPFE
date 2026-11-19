#ifndef CUDAPFE_VEC_POINTS_CUH
#define CUDAPFE_VEC_POINTS_CUH

#include <cstddef>
#include <utility>
#include <cudapfe/vec.hpp>
#include "curve/curve.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/reduction.hpp"
#include "vec/storage.hpp"

// Jacobian-to-affine plumbing shared by the translation units that produce point vectors (vec.cu, scaled.cu,
// msm_g1.cu, msm_g2.cu, mul_generator.cu). They are split so that their heavy device code compiles in parallel.
namespace cudapfe::detail{
    inline constexpr std::size_t kAffineChunk = 8;

    template <class F>
    struct ToAffineOp{
        const Jacobian<F>* in;
        Chunks chunks;
        Affine<F>* out;

        CUDAPFE_HD void operator()(const std::size_t i) const{
            const auto chunk = chunks.at(i);
            to_affine(in + chunk.first, chunk.count, out + chunk.first);
        }
    };

    struct JacobianSum{
        template <class F>
        CUDAPFE_HD Jacobian<F> operator()(const Jacobian<F>& p, const Jacobian<F>& q) const{ return add(p, q); }
    };

    template <class G, Engine E>
    Vec<G, E> normalized(const Buffer<Jacobian<FieldOf<G>>, E>& points){
        Buffer<ElementOf<G>, E> out(points.size());
        const Chunks chunks{1, points.size(), kAffineChunk};
        for_each<E>(chunks.count(), ToAffineOp<FieldOf<G>>{points.data(), chunks, out.data()});
        finish<E>();
        return vec<G, E>(std::move(out));
    }

    template <class G, Engine E, class MakeOp>
    Vec<G, E> generate_points(const std::size_t size, const MakeOp& make_op){
        Buffer<Jacobian<FieldOf<G>>, E> points(size);
        for_each<E>(size, make_op(points.data()));
        return normalized<G, E>(points);
    }

    // Defined in scaled.cu for G1 and G2.
    template <class G, Engine E>
    Vec<G, E> scale_points(const Vec<G, E>& points, const Vec<Zp, E>& k);
}

#endif
