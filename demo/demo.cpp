#include <iostream>
#include <cudapfe/cudapfe.hpp>

namespace{
    template <cudapfe::Engine E>
    bool pairs_to_inner_product(const cudapfe::Vector& x, const cudapfe::Vector& y){
        using namespace cudapfe;
        const auto ps = mul_generator<G1>(Vec<Zp, E>::upload(x));
        const auto qs = mul_generator<G2>(Vec<Zp, E>::upload(y));
        return pair_segments(ps, qs, PairShape{1, x.size()}).at(0) == Gt::generator().pow(inner(x, y));
    }
}

int main(){
    using namespace cudapfe;
    const auto x = random_vector(10);
    const auto y = random_vector(10);

    const bool ok = gpu_available() ? pairs_to_inner_product<Gpu>(x, y) : pairs_to_inner_product<Cpu>(x, y);
    std::cout << (ok ? "BLS12-381: pairing successful" : "BLS12-381: pairing failed") << std::endl;
    return ok ? 0 : 1;
}
