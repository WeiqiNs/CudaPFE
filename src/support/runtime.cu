#include <string>
#include <cufe/engine.hpp>
#include "support/runtime.hpp"

namespace cufe{
    namespace detail{
        void check(const cudaError_t status, const std::string_view operation){
            if (status == cudaSuccess) return;
            cudaGetLastError();
            throw DeviceError(std::string(operation) + ": " + cudaGetErrorString(status));
        }

        GpuRuntime::GpuRuntime(){
            int devices = 0;
            check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
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
}
