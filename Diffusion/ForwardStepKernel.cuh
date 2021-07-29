#pragma once

#include <Array/DeviceArray.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki {
	template<typename T, unsigned REPEAT>
	__global__ void forward_step_kernel(DeviceArray<T,2u> a, DeviceArray<T,2u> const f, DeviceArray<T,1u> const steps) {
		//number of the line to solve
		unsigned ln_idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (0 == ln_idx || a.x_size - 1 <= ln_idx) return;

		//index of the element in a line
		unsigned el_idx_begin = blockIdx.y * blockDim.y + threadIdx.y;

		for (unsigned el_cnt = 0; el_cnt != REPEAT; ++el_cnt) {
			auto el_idx = el_idx_begin + blockDim.y * el_cnt;
			if (0 == el_idx || a.y_size - 1 <= el_idx) continue;

			a(el_idx, ln_idx) = 2 / (steps(el_idx - 1) * (steps(el_idx - 1) + steps(el_idx));
		}
	}
}/*iki*/