#include <cstddef>
#include <string>
#include <cudapfe/engine.hpp>
#include "support/runtime.hpp"

namespace cudapfe{
    namespace{
        __global__ void probe_kernel_image(){}
    }

    namespace detail{
        void check(const cudaError_t status, const std::string_view operation){
            if (status == cudaSuccess) return;
            cudaGetLastError();
            throw DeviceError(std::string(operation) + ": " + cudaGetErrorString(status));
        }

        GpuRuntime::GpuRuntime(){
            int devices = 0;
            check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
            int device = 0;
            check(cudaGetDevice(&device), "cudaGetDevice");
            check(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device), "cudaDeviceGetAttribute");
            cudaFuncAttributes attributes{};
            const auto image = cudaFuncGetAttributes(&attributes, probe_kernel_image);
            if (image == cudaErrorNoKernelImageForDevice || image == cudaErrorInvalidDeviceFunction){
                cudaGetLastError();
                int major = 0;
                int minor = 0;
                check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
                    "cudaDeviceGetAttribute");
                check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device),
                    "cudaDeviceGetAttribute");
                throw DeviceError("this build has no kernel image for compute capability " + std::to_string(major) + "."
                    + std::to_string(minor) + "; reconfigure with -DCMAKE_CUDA_ARCHITECTURES=" + std::to_string(major)
                    + std::to_string(minor));
            }
            check(image, "cudaFuncGetAttributes");
            check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        }

        GpuRuntime::~GpuRuntime(){ cudaStreamDestroy(stream_); }

        void GpuRuntime::synchronize() const{
            check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        }

        const GpuRuntime& GpuRuntime::require(){
            static const GpuRuntime runtime;
            return runtime;
        }
    }

    bool gpu_available(){
        try{
            detail::GpuRuntime::require();
            return true;
        } catch (const DeviceError&){
            return false;
        }
    }

    std::size_t gpu_free_memory(){
        detail::GpuRuntime::require();
        std::size_t free = 0;
        std::size_t total = 0;
        detail::check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
        return free;
    }
}
