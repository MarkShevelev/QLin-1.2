#pragma once

#include <vector>
#include <array>
#include <utility>

namespace iki {
	template <typename T, size_t Dim>
	class HostArray final { };

    template <typename T>
    class HostArray<T,1u> final {
    private:
        size_t size;
        std::vector<T> h_mem;

    public:
        HostArray(size_t size) : size(size), h_mem(size) { }

        size_t get_size() const {
            return size;
        }

        T const *data() const {
            return h_mem.data();
        }

        T *data() {
            return h_mem.data();
        }

        T operator()(size_t idx) const {
            return h_mem[idx];
        }

        T& operator()(size_t idx) {
            return h_mem[idx];
        }

        size_t get_full_size() const {
            return size;
        }
    };

    template <typename T>
    class HostArray<T, 2u> final {
    private:
        size_t y_size, x_size
        std::vector<T> h_mem;

    public:
        HostArray(size_t y_size, size_t x_size) : y_size(y_size), x_size(x_size), h_mem(y_size * x_size) { }

        size_t get_ysize() const {
            return y_size;
        }
        
        size_t get_xsize() const {
            return x_size;
        }

        T const *data() const {
            return h_mem.data();
        }

        T *data() {
            return h_mem.data();
        }

        T operator()(size_t y_idx, size_t x_idx) const {
            return host_memory[y_idx * x_size + x_idx];
        }

        T &operator()(size_t y_idx, size_t x_idx) {
            return host_memory[y_idx * x_size + x_idx];
        }

        void swap_sizes() {
            std::swap(y_size, x_size);
        }

        size_t get_full_size() const {
            return y_size * x_size;
        }
    };
}/*iki*/