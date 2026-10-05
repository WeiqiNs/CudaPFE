#ifndef CUFE_SUPPORT_RUNTIME_HPP
#define CUFE_SUPPORT_RUNTIME_HPP

#include <string_view>
#include <cuda_runtime.h>
#include <cufe/errors.hpp>

namespace cufe::detail{
    void check(cudaError_t status, std::string_view operation);

    class GpuRuntime{
    public:
        GpuRuntime(const GpuRuntime&) = delete;
        GpuRuntime& operator=(const GpuRuntime&) = delete;
        ~GpuRuntime();

        static const GpuRuntime& require();

        [[nodiscard]] cudaStream_t stream() const{ return stream_; }

    private:
        GpuRuntime();

        cudaStream_t stream_ = nullptr;
    };
}

#endif
