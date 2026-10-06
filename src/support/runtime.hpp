#ifndef CUDAPFE_SUPPORT_RUNTIME_HPP
#define CUDAPFE_SUPPORT_RUNTIME_HPP

#include <string_view>
#include <cuda_runtime.h>
#include <cudapfe/errors.hpp>

namespace cudapfe::detail{
    void check(cudaError_t status, std::string_view operation);

    class GpuRuntime{
    public:
        GpuRuntime(const GpuRuntime&) = delete;
        GpuRuntime& operator=(const GpuRuntime&) = delete;
        ~GpuRuntime();

        static const GpuRuntime& require();

        [[nodiscard]] cudaStream_t stream() const{ return stream_; }

        [[nodiscard]] int sm_count() const{ return sm_count_; }

        void synchronize() const;

    private:
        GpuRuntime();

        cudaStream_t stream_ = nullptr;
        int sm_count_ = 0;
    };
}

#endif
