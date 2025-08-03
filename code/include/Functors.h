#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <sycl/sycl.hpp>
#include <ENVI_reader.h>

namespace Functors {
    template<typename Data_access>
    struct BaseFunctor {
        Data_access img_d;
        Data_access spectrums_d;
        Data_access results_d;
        size_t n_spectrums;
        size_t n_lines;
        size_t n_cols;
        size_t bands_size;
        bool local_memory = false;
        sycl::local_accessor<float, 1>* local_data;
        size_t coalesced_memory_width = 1;

        //BaseFunctor() = default;
        BaseFunctor(Data_access img_in, Data_access spectrums_in, Data_access results_in, size_t n_spectrums_in, size_t n_lines_in, size_t n_cols_in, size_t bands_size_in) 
                : img_d(img_in), spectrums_d(spectrums_in), results_d(results_in), n_spectrums(n_spectrums_in), n_lines(n_lines_in), n_cols(n_cols_in), bands_size(bands_size_in) {}
                
        BaseFunctor(Data_access img_in, Data_access spectrums_in, Data_access results_in, size_t n_spectrums_in, size_t n_lines_in, size_t n_cols_in, size_t bands_size_in, sycl::local_accessor<float, 1>* local_data_in, size_t coalesced_memory_width_in) 
                : BaseFunctor(img_in, spectrums_in, results_in, n_spectrums_in, n_lines_in, n_cols_in, bands_size_in)
                {
                    local_memory = true;
                    local_data = local_data_in;
                    coalesced_memory_width = coalesced_memory_width_in;
                }
    };
    template<typename Data_access>
    BaseFunctor(Data_access, Data_access, Data_access, size_t, size_t,size_t) -> BaseFunctor<Data_access>;
    template<typename Data_access>
    BaseFunctor(Data_access, Data_access, Data_access, size_t, size_t,size_t, sycl::local_accessor<float, 1>, size_t) -> BaseFunctor<Data_access>;


    template<typename Data_access>
    struct ImgScaler {
        Data_access img_d;
        int scale_factor;

        ImgScaler(Data_access img_in, int reflectance_scale_factor) : scale_factor(reflectance_scale_factor/100), img_d(img_in) {}

        void operator()(sycl::id<1> i) const { img_d[i] /= scale_factor; }
    };
    template<typename Data_access>
    ImgScaler(Data_access, int) -> ImgScaler<Data_access>;
    

    template<typename Data_access>
    struct Euclidean : BaseFunctor<Data_access>{  //there is no need to calculate the sqrt because we can compare the distances without it, saving computation :)

        //nd kernel with local memory
        Euclidean(Data_access img_in, 
                  Data_access spectrums_in, 
                  Data_access results_in, 
                  size_t n_spectrums_in,
                  size_t n_lines_in,
                  size_t n_cols_in,
                  size_t bands_size_in, 
                  sycl::local_accessor<float, 1> local_data_in, 
                  size_t coalesced_memory_width_in, 
                  size_t sums_per_work_item_in) 
            : BaseFunctor<Data_access>(img_in, spectrums_in, results_in, n_spectrums_in, n_lines_in, n_cols_in, bands_size_in, local_data_in, coalesced_memory_width_in) {}

        //basic kernel and nd-kernel without local memory
        Euclidean(Data_access img_in, 
                  Data_access spectrums_in, 
                  Data_access results_in, 
                  size_t n_spectrums_in,
                  size_t n_lines_in,
                  size_t n_cols_in,
                  size_t bands_size_in) 
            : BaseFunctor<Data_access>(img_in, spectrums_in, results_in, n_spectrums_in, n_lines_in, n_cols_in, bands_size_in) {}

        static size_t get_results_size(size_t image_2D_size, size_t n_bands, size_t n_spectrums) { return image_2D_size * 2; }
        static int get_range_size(size_t image_2D_size, size_t n_bands, size_t n_spectrums) { return image_2D_size * n_spectrums; }

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
            
            //this->results_d[wi_id] = sum;
        }

        /**
         * @brief nd-kernel
         * 
         * Supports local memory, the local memory size has to be (bands_size + (get_local_range(0) * n_spectrums))
         */
        void operator()(sycl::nd_item<1> id) const {
            int id_local = id.get_local_id(0);
            short n_group_ids = id.get_local_range(0);
            short reads_per_work_item = this->bands_size / n_group_ids;

            size_t global_start = id.get_group(0) * this->bands_size + id_local;

            if(this->local_memory) {
                int end = sycl::min(id_local + (reads_per_work_item * this->coalesced_memory_width), this->bands_size - 1);

                for(int index = id_local; index < end; index += this->coalesced_memory_width)     //coalesced read
                    this->local_data[index] = this->img_d[global_start + index];

                int spectrum_offset, partial_spectrum_summation_offset;
                int work_item_result_offset = id_local * this->n_spectrums;
                float diff;
                for(int spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                    spectrum_offset = spectrum * this->bands_size;  //spectrum that is being compared
                    partial_spectrum_summation_offset = this->bands_size + spectrum * n_group_ids + id_local;  //where the partial sum of the work item will be saved

                    for(int index = id_local; id_local < end; index += this->coalesced_memory_width) {
                        diff = this->spectrums_d[spectrum_offset + index] - this->local_data[index];
                        this->local_data[partial_spectrum_summation_offset] += diff * diff;
                    }
                }
                id.barrier(sycl::access::fence_space::local_space);   //wait till all partial distances are calculated

                for(int summation_index = id_local; id_local < this->n_spectrums - 1; summation_index++) {      //1 work-item per spectrum, if n_spectrums>n_group_ids iterate
                    int total_summation_start = this->bands_size + (summation_index * n_group_ids);
                    int total_summation_end = total_summation_start + n_group_ids;
                    for(int total_summation_index = total_summation_start; total_summation_index < total_summation_end; total_summation_index++) {
                        this->local_data[total_summation_start] += this->local_data[total_summation_index];
                    }
                }
                id.barrier(sycl::access::fence_space::local_space);     //wait till all total distances are calculated

                if(id_local == 0) {     //1 work item reduces the result
                    float min_distance = 3.4028235e+38;  //FLOAT_MAX
                    int nearest = 0;
                    float distance;

                    for(int spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                        distance = this->local_data[this->bands_size + spectrum * n_group_ids];
                        if(distance < min_distance) {
                            min_distance = distance;
                            nearest = spectrum;
                        }
                    }

                    this->results_d[id.get_global_id() / n_group_ids] = nearest;
                }
            }   //end if for local memory

            else {      //nd-kernel with no local memory
                int global_end = sycl::min(global_start + (reads_per_work_item * this->coalesced_memory_width), global_start + this->bands_size - 1);
                int global_pixel_index = id.get_global_id() + this->bands_size;

                float distance, lowest_distance = 3.4028235e+38;  //FLOAT_MAX
                int nearest = 0;
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> summation(this->results_d[global_pixel_index]);
                for(int spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                    distance = 0.f;
                    for(int index = global_start; index < global_end; index += this->coalesced_memory_width)     //coalesced read
                        distance += this->img_d[global_start + index];

                    summation.fetch_add(distance);

                    if(id_local == 0) {
                        if(this->results_d[global_pixel_index] < lowest_distance) {
                            lowest_distance = this->results_d[global_pixel_index];
                            nearest = spectrum;
                        }
                    }
                    id.barrier(sycl::access::fence_space::local_space);
                }

                if(id_local == 0)
                    this->results_d[global_pixel_index] = nearest;
            }        
        }
    };
    template<typename Data_access>
    Euclidean(Data_access, Data_access, Data_access, sycl::local_accessor<float, 1>, size_t, size_t, size_t, size_t, size_t) -> Euclidean<Data_access>;
    template<typename Data_access>
    Euclidean(Data_access, Data_access, Data_access, size_t, size_t, size_t, size_t) -> Euclidean<Data_access>;
};

#endif
