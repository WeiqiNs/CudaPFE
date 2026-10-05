#ifndef CUFE_TEST_FE_SCHEMES_HPP
#define CUFE_TEST_FE_SCHEMES_HPP

#include <cstddef>
#include <string_view>
#include <cufe/fe/ipfe_bjk.hpp>
#include <cufe/fe/ipfe_kim.hpp>
#include <cufe/fe/ipfe_kks.hpp>
#include <cufe/fe/ipfe_lin.hpp>
#include <cufe/fe/ipfe_opt.hpp>
#include <cufe/fe/ipfe_tao.hpp>

template <cufe::Engine E>
auto table_decryptor(const cufe::Gt& base, const cufe::Range& range){
    return [table = cufe::DlogTable<E>(base, range)](const auto& sk, const auto& ct){ return dec(table, sk, ct); };
}

inline auto range_decryptor(const cufe::Range& range){
    return [range](const auto& sk, const auto& ct){ return dec(sk, ct, range); };
}

template <cufe::Engine E>
struct Bjk{
    using Engine = E;
    static constexpr std::string_view name = "Bishop et al.";
    static auto setup(const std::size_t n){ return cufe::IPFE::BJK::setup<E>(n); }
    static auto decryptor(const cufe::IPFE::BJK::Msk<E>&, const cufe::Range& range){ return range_decryptor(range); }
};

template <cufe::Engine E>
struct Tao{
    using Engine = E;
    static constexpr std::string_view name = "Tomida et al.";
    static auto setup(const std::size_t n){ return cufe::IPFE::TAO::setup<E>(n); }
    static auto decryptor(const cufe::IPFE::TAO::Msk<E>& msk, const cufe::Range& range){
        return table_decryptor<E>(msk.base, range);
    }
};

template <cufe::Engine E>
struct Kim{
    using Engine = E;
    static constexpr std::string_view name = "Kim et al.";
    static auto setup(const std::size_t n){ return cufe::IPFE::KIM::setup<E>(n); }
    static auto decryptor(const cufe::IPFE::KIM::Msk<E>&, const cufe::Range& range){ return range_decryptor(range); }
};

template <cufe::Engine E>
struct Lin{
    using Engine = E;
    static constexpr std::string_view name = "Lin";
    static auto setup(const std::size_t n){ return cufe::IPFE::LIN::setup<E>(n); }
    static auto decryptor(const cufe::IPFE::LIN::Msk<E>&, const cufe::Range& range){
        return table_decryptor<E>(cufe::IPFE::LIN::base(), range);
    }
};

template <cufe::Engine E>
struct Kks{
    using Engine = E;
    static constexpr std::string_view name = "Kim, Kim and Seo";
    static auto setup(const std::size_t n){ return cufe::IPFE::KKS::setup<E>(n); }
    static auto decryptor(const cufe::IPFE::KKS::Msk<E>&, const cufe::Range& range){
        return table_decryptor<E>(cufe::IPFE::KKS::base(), range);
    }
};

template <cufe::Engine E>
struct Opt{
    using Engine = E;
    static constexpr std::string_view name = "Ojaswi et al.";
    static auto setup(const std::size_t n){ return cufe::IPFE::OPT::setup<E>(n); }
    static auto decryptor(const cufe::IPFE::OPT::Msk<E>&, const cufe::Range& range){
        return table_decryptor<E>(cufe::IPFE::OPT::base(), range);
    }
};

#endif
