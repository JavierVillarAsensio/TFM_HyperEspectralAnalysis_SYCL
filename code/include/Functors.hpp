#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <sycl/sycl.hpp>

namespace Functors {
    template<typename Data_access, bool use_local_memory>
    struct BaseFunctor {
        Data_access img_d, spectrums_d, results_d;
        size_t n_spectrums, n_lines, n_cols, bands_size;
        size_t coalesced_memory_width = 1;

        BaseFunctor(Data_access img_in, Data_access spectrums_in, Data_access results_in, size_t n_spectrums_in, size_t n_lines_in, size_t n_cols_in, size_t bands_size_in, size_t coalesced_memory_width_in) 
                : img_d(img_in),
                  spectrums_d(spectrums_in),
                  results_d(results_in),
                  n_spectrums(n_spectrums_in),
                  n_lines(n_lines_in),
                  n_cols(n_cols_in),
                  bands_size(bands_size_in),
                  coalesced_memory_width(coalesced_memory_width_in) {}

        BaseFunctor() = default;

        static inline constexpr size_t get_n_access_points() { return 3; }
        static inline const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return 0; }
        static inline const size_t get_range_local_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return 0; }
        static inline const size_t get_results_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool nd) { return 0; }
        static inline const size_t get_local_mem_size() { return 0; }
        static inline constexpr bool has_ND() { return true;}

    };

    template<typename Data_access, bool use_local_memory>
    struct ImgScaler : BaseFunctor<Data_access, use_local_memory>{
        Data_access img_d;
        int scale_factor;

        ImgScaler(Data_access img_in, int reflectance_scale_factor) : scale_factor(reflectance_scale_factor/100), img_d(img_in) {}

        inline static constexpr size_t get_n_access_points() { return 1; }
        inline static const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return lines * cols * n_spectrums;}
        inline static const size_t get_results_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool nd) { return lines * cols; }
        inline static constexpr bool has_ND() { return false; }

        void operator()(sycl::id<1> i) const { img_d[i] /= scale_factor; } 
    };    

    template<typename Data_access, bool use_local_memory>
    struct Euclidean : BaseFunctor<Data_access, use_local_memory>{  //there is no need to calculate the sqrt because we can compare the distances without it, saving computation :)

        Euclidean(Data_access img_in, 
                  Data_access spectrums_in, 
                  Data_access results_in, 
                  size_t n_spectrums_in,
                  size_t n_lines_in,
                  size_t n_cols_in,
                  size_t bands_size_in,
                  size_t coalesced_memory_width_in)
            : BaseFunctor<Data_access, use_local_memory>(img_in, spectrums_in, results_in, n_spectrums_in, n_lines_in, n_cols_in, bands_size_in, coalesced_memory_width_in) {}

        inline static size_t get_results_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool nd) { 
            if(nd)
                return lines * cols;
            else
                return lines * cols * 2; 
        }

        inline static size_t get_range_global_size(size_t lines, size_t cols, size_t n_bands, size_t n_spectrums, bool has_local_mem) { 
            if(has_local_mem)
                return lines * cols * n_spectrums; 
            else
                return lines * cols * n_spectrums;
        }

        inline static size_t get_range_local_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { 
            if(has_local_mem)
                return cols; 
            else
                return n_spectrums;
        }

        inline const size_t get_local_mem_size() { return (this->n_cols * this->bands_size * sizeof(float)) + (this->n_spectrums * this->bands_size * sizeof(float)); }
            
        //kernel for basic
        void operator()(sycl::id<1> id) const {
            
            size_t wi_id = id[0];
            
            size_t img_2D_size = this->n_lines * this->n_cols;

            //pixel to be compared
            size_t pixel_offset = wi_id % img_2D_size;
            
            //spectrum to be compared
            size_t spectrum_offset = (wi_id / img_2D_size) * this->bands_size;

            //bil img offset            line                        line size                      line offset
            size_t img_offset = (((wi_id / this->n_cols) * (this->n_cols * this->bands_size)) + (wi_id % this->n_cols)) % (img_2D_size * this->bands_size);
            
            float sum = 0.f, diff;
            for(int i = 0; i < this->bands_size; i++) {
                diff = this->img_d[img_offset + (i * this->n_cols)] - this->spectrums_d[spectrum_offset + i];
                sum += diff * diff;
            }
            
            //                                                                                                                                                where the lowest value is stored
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> lowest_distance(this->results_d[img_2D_size + pixel_offset]);
            float read_distance = lowest_distance.load();
            while(std::fabs(sum) < std::fabs(read_distance)) {
                if(lowest_distance.compare_exchange_weak(read_distance, sum, sycl::memory_order::relaxed)) {   //compare only if "read_distance" remains as the stored value, if so, change it
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> nearest_spectrum(this->results_d[pixel_offset]);
                    nearest_spectrum.store(spectrum_offset / this->bands_size);
                    break;
                }
                read_distance = lowest_distance.load();
            }
        }
        
        void operator()(sycl::nd_item<1> id) const {
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();

            //bil img offset              line                             line size                        group sample
            size_t img_offset = ((group_id / this->n_cols) * (this->n_cols * this->bands_size)) + (group_id % this->n_cols);

            size_t spectrum_offset = local_id * this->bands_size;

            float sum = 0.f, diff;
            for(int i = 0; i < this->bands_size; i++) {
                diff = this->img_d[img_offset + (i * this->n_cols)] - this->spectrums_d[spectrum_offset + i];
                sum += diff * diff;
            }
            
            //                                                                                                                                       where the lowest value is stored
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> lowest_distance(this->results_d[group_id]);
            float read_distance = lowest_distance.load();
            while(std::fabs(sum) < std::fabs(read_distance))
                lowest_distance.compare_exchange_weak(read_distance, sum, sycl::memory_order::relaxed);

            id.barrier();   //the lowest value is already in the results
            
            read_distance = this->results_d[group_id];
            if(std::fabs(sum - read_distance) < 0.0001)
                lowest_distance.store(local_id);    //store the index of the nearest spectrum  
        }
        
        /*void operator()(sycl::nd_item<1> id, sycl::local_accessor<float, 1> local_data) const {
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();
            size_t group_range = id.get_group_range()[0];

            size_t result_pixel_index = (group_id * group_range) + local_id;

            //bil img offset              line                          line size                     group 1st sample
            size_t img_offset = group_id * (group_range * this->bands_size);

            size_t coalesced_read_start = img_offset + local_id;
            size_t coalesced_read_end = img_offset + (group_range * this->bands_size);
            size_t local_data_store_index = local_id * this->bands_size;

            local_data[0] = 0;

            for(size_t global_read_index = coalesced_read_start ; global_read_index < coalesced_read_end; global_read_index+= this->bands_size)
                (*this->local_data)[local_data_store_index++] = this->img_d[global_read_index];


            float sum, diff;
            size_t lowest_index = 0;
            size_t bands_index_start = local_id * this->bands_size;
            size_t bands_index_end = (local_id + 1) * this->bands_size;
            size_t spectrum_band_index;

            for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                sum = 0.f;
                spectrum_band_index = spectrum * this->bands_size;

                for(size_t bands_index = bands_index_start; bands_index < bands_index_end; bands_index++) {
                    diff = (*this->local_data)[bands_index] - this->spectrums_d[spectrum_band_index++];
                    sum += diff * diff;
                }

                if(this->results_d[result_pixel_index] > sum) {
                    this->results_d[result_pixel_index] = sum;
                    lowest_index = spectrum;
                }
            }

            this->results_d[result_pixel_index] = lowest_index;
        }*/
    };
};

#endif
