#pragma once

#include <Array/DeviceArray.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki {
	template<typename T>
	__global__ void forward_step_abc_kernel(
		DeviceArray<T,2u> a, 
		DeviceArray<T,2u> b, 
		DeviceArray<T,2u> c, 
		DeviceArray<T,2u> const f, 
		DeviceArray<T,2u> const dfc_along, 
		DeviceArray<T,1u> const steps, 
		T dt, 
		size_t const repeat
) {
		//number of the line to solve
		unsigned ln_idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (0 == ln_idx || a.x_size - 1 <= ln_idx) return;

		//index of the element in a line
		unsigned el_idx_begin = blockIdx.y * blockDim.y * repeat + threadIdx.y;

		for (unsigned el_cnt = 0; el_cnt != repeat; ++el_cnt) {
			auto el_idx = el_idx_begin + blockDim.y * el_cnt;
			if (0 == el_idx || a.y_size - 1 <= el_idx) continue;

			auto left = dt * dfc_along(el_idx - 1, ln_idx) / steps(el_idx - 1) / (steps(el_idx - 1) + steps(el_idx));
			auto right = dt * dfc_along(el_idx, ln_idx) / steps(el_idx) / (steps(el_idx - 1) + steps(el_idx));
			
			a(el_idx, ln_idx) = -left;
			c(el_idx, ln_idx) = -right;
			b(el_idx, ln_idx) = T(1) + (left + right);
		}
	}

	template<typename T>
	__global__ void forward_step_d_kernel(
		DeviceArray<T, 2u> d,
		DeviceArray<T, 2u> const f_curr,
		DeviceArray<T, 2u> const dfc_along,
		DeviceArray<T, 2u> const dfc_perp,
		DeviceArray<T, 1u> const steps_along,
		DeviceArray<T, 1u> const steps_perp,
		T dt,
		size_t const repeat
	) { 
		//number of the line to solve
		unsigned ln_idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (0 == ln_idx || a.x_size - 1 <= ln_idx) return;

		//index of the element in a line
		unsigned el_idx_begin = blockIdx.y * blockDim.y * repeat + threadIdx.y;

		for (unsigned el_cnt = 0; el_cnt != repeat; ++el_cnt) {
			auto el_idx = el_idx_begin + blockDim.y * el_cnt;
			if (0 == el_idx || a.y_size - 1 <= el_idx) continue;

			auto left = dt * dfc_along(el_idx - 1, ln_idx) / steps_along(el_idx - 1) / (steps_along(el_idx - 1) + steps_along(el_idx));
			auto right = dt * dfc_along(el_idx, ln_idx) / steps_along(el_idx) / (steps_along(el_idx - 1) + steps_along(el_idx));

			auto up = 2 * dt * dfc_perp(el_idx, ln_idx - 1) / steps_perp(ln_idx - 1) / (steps_perp(ln_idx - 1) + steps_perp(ln_idx));
			auto down = 2 * dt * dfc_perp(el_idx, ln_idx) / steps_perp(ln_idx) / (steps_perp(ln_idx - 1) + steps_perp(ln_idx));

			d(el_idx, ln_idx) = f_curr(el_idx, ln_idx)
				+ left * f_curr(el_idx - 1, ln_idx)
				+ right * f_curr(el_idx + 1, ln_idx)
				- (left + right) * f_curr(el_idx, ln_idx)
				+ up * f_curr(el_idx, ln_idx - 1)
				+ down * f_curr(el_idx, ln_idx + 1)
				- (up + down) * f_curr(el_idx, ln_idx);
		}
	}
}/*iki*/