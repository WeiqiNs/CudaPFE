#ifndef CUDAPFE_ERRORS_HPP
#define CUDAPFE_ERRORS_HPP

#include <stdexcept>

namespace cudapfe{
    class Error : public std::runtime_error{
    public:
        using std::runtime_error::runtime_error;
    };

    class DecodeError : public Error{
    public:
        using Error::Error;
    };

    class ShapeError : public Error{
    public:
        using Error::Error;
    };

    class NotInvertible : public Error{
    public:
        using Error::Error;
    };

    class DeviceError : public Error{
    public:
        using Error::Error;
    };
}

#endif
