#pragma once

#include <Array/DeviceArray.cuh>

namespace iki {
	template <typename T>
	struct DeviceNonuniformGrid {
		__device__ __host__ DeviceNonuniformGrid(DeviceArray<T, 1u> steps, DeviceArray<T, 1u> values): steps(steps), values(values) { }
		DeviceArray<T, 1u> const steps;
		DeviceArray<T, 1u> const values;
	};
}/*iki*/