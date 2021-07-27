#pragma once

#include <Array/DeviceArray.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki {
	template<typename T>
	__global__ void thomson_sweep_kernel(
		iki::DeviceArray<T, 2u> a,
		iki::DeviceArray<T, 2u> b,
		iki::DeviceArray<T, 2u> c,
		iki::DeviceArray<T, 2u> d,
		iki::DeviceArray<T, 2u> res
	) {
		size_t x_size = res.x_size, y_size = res.y_size;
		size_t x_idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (x_idx == 0 || x_idx >= x_size - 1) return;

		for (size_t y_idx = 2; y_idx != y_size - 1; ++y_idx) {
			T w = a(y_idx, x_idx) / b(y_idx - 1, x_idx);
			b(y_idx, x_idx) = fma(-w, c(y_idx - 1, x_idx), b(y_idx, x_idx));
			d(y_idx, x_idx) = fma(-w, d(y_idx - 1, x_idx), d(y_idx, x_idx));
		}
		res(y_size - 2, x_idx) = d(y_size - 2, x_idx) / b(y_size - 2, x_idx);

		for (size_t y_idx = y_size - 3; y_idx != 0; --y_idx) {
			res(y_idx, x_idx) = fma(-c(y_idx, x_idx), res(y_idx + 1, x_idx), d(y_idx, x_idx)) / b(y_idx, x_idx);
		}
	}
}/*iki*/