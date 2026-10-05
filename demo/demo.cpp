#include <iostream>
#include <cufe/cufe.hpp>

int main(){
    using namespace cufe;
    const auto x = random_vector(10);
    const auto y = random_vector(10);

    const bool ok = pair(G1::mul_generator(x), G2::mul_generator(y)) == Gt::generator().pow(inner(x, y));
    std::cout << (ok ? "BLS12-381: pairing successful" : "BLS12-381: pairing failed") << std::endl;
    return ok ? 0 : 1;
}
