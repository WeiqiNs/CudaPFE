#ifndef CUDAPFE_TEST_FE_SCHEMES_HPP
#define CUDAPFE_TEST_FE_SCHEMES_HPP

#include <cstddef>
#include <string_view>
#include <cudapfe/fe/ipfe_bjk.hpp>
#include <cudapfe/fe/ipfe_kim.hpp>
#include <cudapfe/fe/ipfe_kks.hpp>
#include <cudapfe/fe/ipfe_lin.hpp>
#include <cudapfe/fe/ipfe_opt.hpp>
#include <cudapfe/fe/ipfe_tao.hpp>
#include <cudapfe/fe/qfe_bcfg.hpp>
#include <cudapfe/fe/qfe_sgp.hpp>

template <cudapfe::Engine E>
auto table_decryptor(const cudapfe::Gt& base, const cudapfe::Range& range){
    return [table = cudapfe::DlogTable<E>(base, range)](const auto& sk, const auto& ct){ return dec(table, sk, ct); };
}

inline auto range_decryptor(const cudapfe::Range& range){
    return [range](const auto& sk, const auto& ct){ return dec(sk, ct, range); };
}

template <cudapfe::Engine E>
struct Bjk{
    using Engine = E;
    static constexpr std::string_view name = "Bishop et al.";
    static auto setup(const std::size_t n){ return cudapfe::IPFE::BJK::setup<E>(n); }
    static auto decryptor(const cudapfe::IPFE::BJK::Msk<E>&, const cudapfe::Range& range){ return range_decryptor(range); }
};

template <cudapfe::Engine E>
struct Tao{
    using Engine = E;
    static constexpr std::string_view name = "Tomida et al.";
    static auto setup(const std::size_t n){ return cudapfe::IPFE::TAO::setup<E>(n); }
    static auto decryptor(const cudapfe::IPFE::TAO::Msk<E>& msk, const cudapfe::Range& range){
        return table_decryptor<E>(msk.base, range);
    }
};

template <cudapfe::Engine E>
struct Kim{
    using Engine = E;
    static constexpr std::string_view name = "Kim et al.";
    static auto setup(const std::size_t n){ return cudapfe::IPFE::KIM::setup<E>(n); }
    static auto decryptor(const cudapfe::IPFE::KIM::Msk<E>&, const cudapfe::Range& range){ return range_decryptor(range); }
};

template <cudapfe::Engine E>
struct Lin{
    using Engine = E;
    static constexpr std::string_view name = "Lin";
    static auto setup(const std::size_t n){ return cudapfe::IPFE::LIN::setup<E>(n); }
    static auto decryptor(const cudapfe::IPFE::LIN::Msk<E>&, const cudapfe::Range& range){
        return table_decryptor<E>(cudapfe::IPFE::LIN::base(), range);
    }
};

template <cudapfe::Engine E>
struct Kks{
    using Engine = E;
    static constexpr std::string_view name = "Kim, Kim and Seo";
    static auto setup(const std::size_t n){ return cudapfe::IPFE::KKS::setup<E>(n); }
    static auto decryptor(const cudapfe::IPFE::KKS::Msk<E>&, const cudapfe::Range& range){
        return table_decryptor<E>(cudapfe::IPFE::KKS::base(), range);
    }
};

template <cudapfe::Engine E>
struct Opt{
    using Engine = E;
    static constexpr std::string_view name = "Ojaswi et al.";
    static auto setup(const std::size_t n){ return cudapfe::IPFE::OPT::setup<E>(n); }
    static auto decryptor(const cudapfe::IPFE::OPT::Msk<E>&, const cudapfe::Range& range){
        return table_decryptor<E>(cudapfe::IPFE::OPT::base(), range);
    }
};

template <cudapfe::Engine E>
struct Bcfg{
    using Engine = E;
    static constexpr std::string_view name = "Baltico et al.";
    static auto setup(const std::size_t n){ return cudapfe::QFE::BCFG::setup<E>(n); }
    static auto decryptor(const cudapfe::QFE::BCFG::Keys<E>& keys, const cudapfe::Range& range){
        const cudapfe::DlogTable<E> table(cudapfe::QFE::BCFG::base(), range);
        return [pk = keys.pk, table](const auto& sk, const auto& ct){ return dec(table, pk, sk, ct); };
    }
};

template <cudapfe::Engine E>
struct Sgp{
    using Engine = E;
    static constexpr std::string_view name = "Dufour-Sans et al.";
    static auto setup(const std::size_t n){ return cudapfe::QFE::SGP::setup<E>(n); }
    static auto decryptor(const cudapfe::QFE::SGP::Keys<E>&, const cudapfe::Range& range){
        return table_decryptor<E>(cudapfe::QFE::SGP::base(), range);
    }
};

#endif
