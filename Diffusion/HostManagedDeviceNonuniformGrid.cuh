#pragma once

#include <Array/HostArray.h>
#include <Array/HostManagedDeviceArray.cuh>
#include <Array/Transfer.cuh>
#include <Diffusion/DeviceNonuniformGrid.cuh>

namespace iki {
	template <typename T>
	struct HostManagedDeviceNonuniformAxis final {
	private:
		HostManagedDeviceArray<T, 1u> d_steps, d_values;

	public:
		HostManagedDeviceNonuniformAxis(HostArray<T, 1u> const &h_steps, HostArray<T, 1u> const &h_values) :d_steps(h_steps.get_size()), d_values(h_values.get_size()) {
			transfer(h_steps, d_steps);
			transfer(h_values, d_values);
		}

		DeviceNonuniformAxis<T> axis() const {
			return DeviceNonuniformAxis<T>(steps.array(), values.array());
		}
	};

	template <typename T, size_t Dim>
	struct HostManagedDeviceNonuniformGrid final { };

	template <typename T>
	struct HostManagedDeviceNonuniformGrid<T, 1u> final {
	private:
		HostManagedDeviceNonuniformAxis<T> d_along;

		HostManagedDeviceNonuniformGrid(HostArray<T, 1u> const &h_steps, HostArray<T, 1u> const &h_values): d_along(h_steps, h_values) { }

		DeviceNonuniformGrid<T, 1u> grid() const {
			return DeviceNonuniformGrid<T, 1u>(d_along.axis());
		}
	};

	template <typename T>
	struct HostManagedDeviceNonuniformGrid<T, 2u> final {
	private:
		HostManagedDeviceNonuniformAxis<T> d_along, d_perp;

	public:
		HostManagedDeviceNonuniformGrid(
			HostArray<T, 1u> const &h_steps_along, HostArray<T, 1u> const &h_values_along, 
			HostArray<T, 1u> const &h_steps_perp, HostArray<T, 1u> const &h_values_perp
		) : d_along(h_steps_along, h_values_along), d_perp(h_steps_perp, h_values_pepr) { }

		DeviceNonuniformGrid<T, 2u> grid() const {
			return DeviceNonuniformGrid<T, 2u>(d_along.axis(), d_perp.axis());
		}
	};

	template <typename T>
	HostArray<T, 1u> calculate_values(T begin, HostArray<T, 1u> const &steps) {
		HostArray<T, 1u> values(steps + 1);
		values(0) = begin;
		T value = begin; //sum
		T correction = T(0);
		T y(0), t(0);
		for (size_t idx = 0; idx != steps.get_size(); ++idx) {
			y = stesps(idx) - correction;
			t = value + y;
			correction = (t - value) - y;
			value = t;
			values(idx + 1) = value;
		}
		return values;
	}
}/*iki*/