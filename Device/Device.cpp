#include <Device/Device.h>
#include <Device/DeviceError.h>

#include <cuda_runtime.h>

using namespace std;

namespace iki {
	Device::Device(int device) {
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaSetDevice(device)))
			throw DeviceError(cudaStatus);
	}

	Device::~Device() noexcept {
		cudaDeviceReset();
	}
} /*iki*/