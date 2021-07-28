#pragma once

#include <cuda_runtime.h>

namespace iki {
	template<typename T, size_t Dim>
	struct DeviceArray final { };

	template<typename T>
	struct DeviceArray<T, 1u> final {
        size_t const size; 
        T *const data; //device pointer

        __host__ __device__ Array(size_t size, T *data) : size(size), data(data) { }

        __host__ __device__ get_full_size() const {
            return size;
        }

        __device__ T operator()(size_t idx) const {
            return data[idx];
        }

        __device__ T &operator()(size_t idx) {
            return data[idx];
        }
	};

    template<typename T>
    struct DeviceArray<T, 2u> final {
        size_t const y_size, const x_size;
        T *const data; //device pointer

        __host__ __device__ Array(size_t y_size, size_t x_size, T *data) : y_size(y_size), x_size(x_size), data(data) { }

        __host__ __device__ get_full_size() const {
            return y_size * x_size;
        }

        __device__ T operator()(size_t y_idx, size_t x_idx) const {
            return data[y_idx * x_size + x_idx];
        }

        __device__ T &operator()(size_t y_idx, size_t x_idx) {
            return data[y_idx * x_size + x_idx];
        }
    };
} /*iki*/