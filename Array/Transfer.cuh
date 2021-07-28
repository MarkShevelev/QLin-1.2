#pragma once

#include <Device/DeviceError.h>
#include <Array/DeviceArray.cuh>
#include <Array/HostArray.h>
#include <Array/HostManagedDeviceArray.cuh>

#include <cuda_runtime.h>

#include <stdexcept>

namespace iki {
	template <typename T, size_t Dim>
	void transfer(iki::HostArray<T,Dim> from, iki::DeviceArray<T,Dim> to) {
		if (from.get_full_size() != to.get_full_size())
			throw std::runtime_error("can't transfer data: arrays of different sizes");
		{
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaMemcpy(to.data, from.data(), from.get_full_size() * sizeof(T), cudaMemcpyHostToDevice)))
				throw iki::DeviceError("can't transfer data from host to device: ", cudaStatus);
		}
	}

	template <typename T, size_t Dim>
	void transfer(iki::HostArray<T, Dim> from, iki::HostManagedDeviceArray<T, Dim> to) {
		if (from.get_full_size() != to.get_full_size())
			throw std::runtime_error("can't transfer data: arrays of different sizes");
		{
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaMemcpy(to.data(), from.data(), from.get_full_size() * sizeof(T), cudaMemcpyHostToDevice)))
				throw iki::DeviceError("can't transfer data from host to device: ", cudaStatus);
		}
	}
}/*iki*/