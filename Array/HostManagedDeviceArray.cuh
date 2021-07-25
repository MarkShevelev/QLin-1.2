#pragma once

#include <Device/DeviceMemory.h>
#include <Device/DeviceError.h>
#include <Array/DeviceArray.cuh>

#include <cuda_runtime.h>

#include <utility>

namespace iki {
    template<typename T, size_t Dim>
    struct HostManagedDeviceArray final { };

    template<typename T>
    struct HostManagedDeviceArray<T, 1u> final {
        size_t size;
        DeviceMemory d_mem; //managed device memory

        HostManagedDeviceArray(size_t size) : size(size), d_mem(size * sizeof(T)) { }

        DeviceArray<T, 1u> array() const {
            return DeviceArray<T,1u>(size, d_mem.as<T>());
        }

        T* data() {
            return d_mem.as<T>();
        }

        T const* data() const {
            return d_mem.as<T>();
        }

        size_t get_size() const {
            return size;
        }
    };

    template<typename T>
    struct HostManagedDeviceArray<T, 2u> final {
        size_t y_size, x_size;
        DeviceMemory d_mem; //managed device memory

        HostManagedDeviceArray(size_t y_size, size_t x_size) : y_size(y_size), x_size(x_size), d_mem(y_size * x_size * sizeof(T)) { }

        DeviceArray<T, 2u> array() const {
            return DeviceArray<T,2u>(y_size, x_size, d_mem.as<T>());
        }

        T *data() {
            return d_mem.as<T>();
        }

        T const *data() const {
            return d_mem.as<T>();
        }

        size_t get_ysize() const {
            return x_size;
        }

        size_t get_xsize() const {
            return x_size;
        }

        void swap_sizes() {
            std::swap(y_size, x_size);
        }
    };
} /*iki*/