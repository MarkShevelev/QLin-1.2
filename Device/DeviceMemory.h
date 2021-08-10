#pragma once

namespace iki {
	class DeviceMemory final {
	public:
		DeviceMemory(size_t byte_size);

		DeviceMemory(DeviceMemory const &src);
		DeviceMemory(DeviceMemory &&src) noexcept;
		DeviceMemory& operator=(DeviceMemory const &src);
		DeviceMemory& operator=(DeviceMemory &&src) noexcept;

		void* get() const {
			return device_ptr;
		}

		template <typename T>
		T* as() const {
			return reinterpret_cast<T*>(device_ptr);
		}

		size_t get_size() const {
			return byte_size;
		}

	private:
		void *device_ptr;
		size_t byte_size;
	};
} /*iki*/