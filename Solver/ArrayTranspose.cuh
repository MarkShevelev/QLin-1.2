#pragma once

#include <Array/DeviceArray.cuh>
#include <Array/HostManagedDeviceArray.cuh>
#include <Solver/TransposeKernel.cuh>

#include <cuda_runtime.h>

namespace iki {
	template<typename T>
	void transpose(
		iki::DeviceArray<T,2u> const from, 
		iki::DeviceArray<T,2u> to
	) {
		dim3 grid(x_size / 32, y_size / 8);
		dim3 block(32,8);
		transpose_kernel<32,8><<<grid,block>>>(to.data, from.data, from.x_size, from.y_size);
	}

	template<typename T>
	void transpose(
		iki::HostManagedDeviceArray<T, 2u> const from,
		iki::HostManagedDeviceArray<T, 2u> to
	) {
		transpose(from.array(), to.array());
	}
}/*iki*/