#include "vec/msm.cuh"

namespace cudapfe{
    template Vec<G2, Cpu> msm<G2, Cpu>(const Vec<G2, Cpu>&, const Vec<Zp, Cpu>&, const MsmShape&);
    template Vec<G2, Gpu> msm<G2, Gpu>(const Vec<G2, Gpu>&, const Vec<Zp, Gpu>&, const MsmShape&);
}
