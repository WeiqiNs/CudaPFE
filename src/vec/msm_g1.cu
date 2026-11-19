#include "vec/msm.cuh"

namespace cudapfe{
    template Vec<G1, Cpu> msm<G1, Cpu>(const Vec<G1, Cpu>&, const Vec<Zp, Cpu>&, const MsmShape&);
    template Vec<G1, Gpu> msm<G1, Gpu>(const Vec<G1, Gpu>&, const Vec<Zp, Gpu>&, const MsmShape&);
}
