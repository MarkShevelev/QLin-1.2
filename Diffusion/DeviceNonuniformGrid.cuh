#pragma once

#include <Array/DeviceArray.cuh>

namespace iki {
	template <typename T>
	struct DeviceNonuniformAxis final {
		__device__ __host__ DeviceNonuniformGrid(DeviceArray<T, 1u> steps, DeviceArray<T, 1u> values): steps(steps), values(values) { }
		DeviceArray<T, 1u> const steps;
		DeviceArray<T, 1u> const values;
	};

	template<typename T, size_t Dim>
	struct DeviceNonuniformGrid final {
	};

	template <typename T>
	struct DeviceNonuniformGrid<T, 1u> {
		DeviceNonuniformGrid(DeviceNonuniformAxis<T> along) : along(along) { }
		DeviceNonuniformAxis along;
	};

	template <typename T>
	struct DeviceNonuniformGrid<T, 2u> {
		DeviceNonuniformGrid(DeviceNonuniformAxis<T> along, DeviceNonuniformAxis<T> perp): along(along), perp(perp) { }
		DeviceNonuniformAxis along, perp;
	};
}/*iki*/