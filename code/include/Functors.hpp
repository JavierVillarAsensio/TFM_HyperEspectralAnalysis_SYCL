#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <sycl/sycl.hpp>

namespace Functors {
    template<typename Data_access, bool use_local_memory>
    struct BaseFunctor {
        Data_access img_d, spectrums_d, results_d;
        size_t n_spectrums, n_lines, n_cols, bands_size;
        sycl::local_accessor<float, 1>* local_data;
        size_t coalesced_memory_width = 1;
                
        BaseFunctor(Data_access img_in, Data_access spectrums_in, Data_access results_in, size_t n_spectrums_in, size_t n_lines_in, size_t n_cols_in, size_t bands_size_in, sycl::local_accessor<float, 1>* local_data_in, size_t coalesced_memory_width_in) 
                : img_d(img_in),
                  spectrums_d(spectrums_in),
                  results_d(results_in),
                  n_spectrums(n_spectrums_in),
                  n_lines(n_lines_in),
                  n_cols(n_cols_in),
                  bands_size(bands_size_in),
                  local_data(local_data_in),
                  coalesced_memory_width(coalesced_memory_width_in) {}

        BaseFunctor() = default;

        static inline constexpr size_t get_n_access_points() { return 3; }
        static inline const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums) { return lines * cols; }
        static inline const size_t get_range_local_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums) { return 0; }
    };

    template<typename Data_access, bool use_local_memory>
    struct ImgScaler : BaseFunctor<Data_access, use_local_memory>{
        Data_access img_d;
        int scale_factor;

        ImgScaler(Data_access img_in, int reflectance_scale_factor, sycl::local_accessor<float, 1>* needed_for_polimorphism) : scale_factor(reflectance_scale_factor/100), img_d(img_in) {}

        inline static constexpr size_t get_n_access_points() { return 1; }
        inline static const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums) { return lines * cols * bands; }

        void operator()(sycl::id<1> i) const { img_d[i] /= scale_factor; } 
    };    

    template<typename Data_access, bool use_local_memory>
    struct Euclidean : BaseFunctor<Data_access, use_local_memory>{  //there is no need to calculate the sqrt because we can compare the distances without it, saving computation :)

        //nd kernel with local memory
        Euclidean(Data_access img_in, 
                  Data_access spectrums_in, 
                  Data_access results_in, 
                  size_t n_spectrums_in,
                  size_t n_lines_in,
                  size_t n_cols_in,
                  size_t bands_size_in,
                  size_t coalesced_memory_width_in,
                  sycl::local_accessor<float, 1>* local_data_in)
            : BaseFunctor<Data_access, use_local_memory>(img_in, spectrums_in, results_in, n_spectrums_in, n_lines_in, n_cols_in, bands_size_in, local_data_in, coalesced_memory_width_in) {}

        inline static size_t get_results_size(size_t image_2D_size, size_t n_bands, size_t n_spectrums) { return image_2D_size * 2; }
        inline static int get_range_size(size_t image_2D_size, size_t n_bands, size_t n_spectrums) { return image_2D_size * n_spectrums; }

        //kernel for basic
        void operator()(sycl::id<1> id) const {
            size_t wi_id = id[0];
            size_t img_2D_size = this->n_lines * this->n_cols;

            //pixel to be compared
            size_t pixel_offset = wi_id % img_2D_size;
            
            //spectrum to be compared
            size_t spectrum_offset = (wi_id / img_2D_size) * this->bands_size;

            //bil img offset
            size_t img_offset = (((this->n_lines * this->bands_size) * (wi_id / this->n_lines)) + (wi_id % this->n_lines)) % (img_2D_size * this->bands_size);
            
            float sum = 0.f, diff;
            for(int i = 0; i < this->bands_size; i++) {
                diff = this->img_d[img_offset + (i * this->n_cols)] - this->spectrums_d[spectrum_offset + i];
                sum += diff * diff;
            }
            
            //                                                                                                                                                    where the lowest value is stored
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> lowest_distance(this->results_d[img_2D_size + pixel_offset]);
            float read_distance = lowest_distance.load();
            while(sum < read_distance) {
                if(lowest_distance.compare_exchange_weak(read_distance, sum, sycl::memory_order::relaxed)) {   //compare only if "read_distance" remains as the stored value, if so, change it
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> nearest_spectrum(this->results_d[pixel_offset]);
                    nearest_spectrum.store(spectrum_offset / this->bands_size);
                    break;
                }
                read_distance = lowest_distance.load();
            }
        }

        template<bool B = use_local_memory, std::enable_if_t<!B, int> = 0>
        void operator()(sycl::nd_item<1> id) const {
            
        }

        template<bool B = use_local_memory, std::enable_if_t<B, int> = 0>
        void operator()(sycl::nd_item<1> id) const {
            
        }
    };
};

#endif
