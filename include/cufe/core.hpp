#ifndef CUFE_CORE_HPP
#define CUFE_CORE_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "errors.hpp"

namespace cufe{
    using Bytes = std::vector<std::uint8_t>;
    using ByteView = std::span<const std::uint8_t>;

    enum class Side{ g1, g2 };

    enum class Encoding{ compressed, uncompressed };

    namespace detail{
        struct Access;

        template <class I>
        concept Character = std::same_as<I, char> || std::same_as<I, wchar_t> || std::same_as<I, char8_t>
            || std::same_as<I, char16_t> || std::same_as<I, char32_t>;
    }

    class Zp{
    public:
        Zp() = default;

        template <std::signed_integral I> requires (!detail::Character<I> && sizeof(I) <= sizeof(std::int64_t))
        Zp(const I value) : Zp(from_signed(value)){}

        template <std::unsigned_integral I>
            requires (!std::same_as<I, bool> && !detail::Character<I> && sizeof(I) <= sizeof(std::uint64_t))
        Zp(const I value) : Zp(from_unsigned(value)){}

        static constexpr std::size_t byte_size = 32;

        [[nodiscard]] static Zp random();
        [[nodiscard]] static Zp from_bytes(ByteView bytes);

        [[nodiscard]] Bytes to_bytes() const;
        [[nodiscard]] std::string to_string() const;
        [[nodiscard]] bool is_zero() const;
        [[nodiscard]] Zp inverse() const;
        [[nodiscard]] Zp pow(std::uint64_t exponent) const;

        friend Zp operator+(const Zp& x, const Zp& y){ return x.plus(y); }
        friend Zp operator-(const Zp& x, const Zp& y){ return x.minus(y); }
        friend Zp operator*(const Zp& x, const Zp& y){ return x.times(y); }
        friend Zp operator/(const Zp& x, const Zp& y){ return x.times(y.inverse()); }
        friend Zp operator-(const Zp& x){ return x.negated(); }
        friend bool operator==(const Zp& x, const Zp& y) = default;

        Zp& operator+=(const Zp& y){ return *this = *this + y; }
        Zp& operator-=(const Zp& y){ return *this = *this - y; }
        Zp& operator*=(const Zp& y){ return *this = *this * y; }
        Zp& operator/=(const Zp& y){ return *this = *this / y; }

    private:
        friend struct detail::Access;

        static Zp from_signed(std::int64_t value);
        static Zp from_unsigned(std::uint64_t value);

        [[nodiscard]] Zp plus(const Zp& y) const;
        [[nodiscard]] Zp minus(const Zp& y) const;
        [[nodiscard]] Zp times(const Zp& y) const;
        [[nodiscard]] Zp negated() const;

        std::array<unsigned long long, 4> words_{};
    };

    template <Side S>
    class Point{
    public:
        Point() = default;

        static constexpr std::size_t compressed_size = S == Side::g1 ? 48 : 96;
        static constexpr std::size_t uncompressed_size = 2 * compressed_size;

        [[nodiscard]] static Point generator();
        [[nodiscard]] static Point random();
        [[nodiscard]] static Point mul_generator(const Zp& scalar);

        template <std::same_as<std::vector<Zp>> V>
        [[nodiscard]] static std::vector<Point> mul_generator(const V& scalars){
            std::vector<Point> points;
            points.reserve(scalars.size());
            for (const auto& scalar : scalars) points.push_back(mul_generator(scalar));
            return points;
        }

        [[nodiscard]] static Point from_bytes(ByteView bytes);

        [[nodiscard]] Bytes to_bytes(Encoding encoding = Encoding::compressed) const;
        [[nodiscard]] bool is_identity() const;

        friend Point operator+(const Point& p, const Point& q){ return p.plus(q); }
        friend Point operator-(const Point& p, const Point& q){ return p.plus(q.negated()); }
        friend Point operator-(const Point& p){ return p.negated(); }
        friend Point operator*(const Point& p, const Zp& k){ return p.times(k); }
        friend Point operator*(const Zp& k, const Point& p){ return p.times(k); }
        friend bool operator==(const Point& p, const Point& q){ return p.equals(q); }

        Point& operator+=(const Point& q){ return *this = *this + q; }
        Point& operator-=(const Point& q){ return *this = *this - q; }
        Point& operator*=(const Zp& k){ return *this = *this * k; }

    private:
        friend struct detail::Access;

        [[nodiscard]] Point plus(const Point& q) const;
        [[nodiscard]] Point times(const Zp& k) const;
        [[nodiscard]] Point negated() const;
        [[nodiscard]] bool equals(const Point& q) const;

        std::array<unsigned long long, S == Side::g1 ? 18 : 36> words_{};
    };

    using G1 = Point<Side::g1>;
    using G2 = Point<Side::g2>;

    class Gt{
    public:
        Gt();

        static constexpr std::size_t byte_size = 576;

        [[nodiscard]] static Gt generator();
        [[nodiscard]] static Gt random();
        [[nodiscard]] static Gt from_bytes(ByteView bytes);

        [[nodiscard]] Bytes to_bytes() const;
        [[nodiscard]] bool is_one() const;
        [[nodiscard]] Gt inverse() const;
        [[nodiscard]] Gt pow(const Zp& exponent) const;

        friend Gt operator*(const Gt& x, const Gt& y){ return x.times(y); }
        friend Gt operator/(const Gt& x, const Gt& y){ return x.times(y.inverse()); }
        friend bool operator==(const Gt& x, const Gt& y) = default;

        Gt& operator*=(const Gt& y){ return *this = *this * y; }
        Gt& operator/=(const Gt& y){ return *this = *this / y; }

    private:
        friend struct detail::Access;

        [[nodiscard]] Gt times(const Gt& y) const;

        std::array<unsigned long long, 72> words_;
    };

    [[nodiscard]] Gt pair(const G1& p, const G2& q);

    [[nodiscard]] Gt pair(const std::vector<G1>& ps, const std::vector<G2>& qs);

    void seed(ByteView bytes);

    template <Side S>
    [[nodiscard]] Point<S> sum(const std::vector<Point<S>>& points){
        Point<S> total;
        for (const auto& p : points) total += p;
        return total;
    }

    [[nodiscard]] inline Bytes bytes_of(const std::string_view text){
        return {text.begin(), text.end()};
    }
}

#endif
