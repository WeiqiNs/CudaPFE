#ifndef CUFE_SUPPORT_DEVICE_ARRAY_HPP
#define CUFE_SUPPORT_DEVICE_ARRAY_HPP

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include "support/runtime.hpp"

namespace cufe::detail{
    template <class T>
    class DeviceArray{
    public:
        explicit DeviceArray(const std::size_t size) : size_(size){
            if (size_ == 0) return;
            stream_ = GpuRuntime::require().stream();
            check(cudaMallocAsync(&data_, size_ * sizeof(T), stream_), "cudaMallocAsync");
        }

        DeviceArray(DeviceArray&& other) noexcept
            : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)), stream_(other.stream_){}

        DeviceArray& operator=(DeviceArray&& other) noexcept{
            std::swap(data_, other.data_);
            std::swap(size_, other.size_);
            std::swap(stream_, other.stream_);
            return *this;
        }

        DeviceArray(const DeviceArray&) = delete;
        DeviceArray& operator=(const DeviceArray&) = delete;

        ~DeviceArray(){
            if (data_ != nullptr && cudaFreeAsync(data_, stream_) != cudaSuccess) cudaGetLastError();
        }

        [[nodiscard]] T* data(){ return data_; }

        [[nodiscard]] const T* data() const{ return data_; }

        [[nodiscard]] std::size_t size() const{ return size_; }

        void copy_from(const std::span<const T> source){
            if (source.size() != size_){
                throw ShapeError("DeviceArray of " + std::to_string(size_) + " elements cannot take "
                    + std::to_string(source.size()));
            }
            if (size_ == 0) return;
            check(cudaMemcpyAsync(data_, source.data(), size_ * sizeof(T), cudaMemcpyHostToDevice, stream_),
                "cudaMemcpyAsync to device");
        }

        [[nodiscard]] std::vector<T> copy_to_host() const{
            std::vector<T> host(size_);
            if (size_ == 0) return host;
            check(cudaMemcpyAsync(host.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream_),
                "cudaMemcpyAsync to host");
            check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
            return host;
        }

        [[nodiscard]] T element(const std::size_t index) const{
            T host;
            check(cudaMemcpyAsync(&host, data_ + index, sizeof(T), cudaMemcpyDeviceToHost, stream_),
                "cudaMemcpyAsync to host");
            check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
            return host;
        }

    private:
        T* data_ = nullptr;
        std::size_t size_;
        cudaStream_t stream_ = nullptr;
    };
}

#endif
