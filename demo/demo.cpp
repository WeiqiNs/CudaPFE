#include <iostream>
#include <cufe/cufe.hpp>

namespace{
    template <cufe::Engine E>
    bool pairs_to_inner_product(const cufe::Vector& x, const cufe::Vector& y){
        using namespace cufe;
        const auto ps = mul_generator<G1>(Vec<Zp, E>::upload(x));
        const auto qs = mul_generator<G2>(Vec<Zp, E>::upload(y));
        return pair_segments(ps, qs, PairShape{1, x.size()}).at(0) == Gt::generator().pow(inner(x, y));
    }
}

int main(){
    using namespace cufe;
    const auto x = random_vector(10);
    const auto y = random_vector(10);

    const bool ok = gpu_available() ? pairs_to_inner_product<Gpu>(x, y) : pairs_to_inner_product<Cpu>(x, y);
    std::cout << (ok ? "BLS12-381: pairing successful" : "BLS12-381: pairing failed") << std::endl;
    return ok ? 0 : 1;
}
