#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <sycl/sycl.hpp>
#include <optional>

#define FLOAT_MAX 3.4028235e+38

template<typename Data_access>
inline float compute_correlation(
    size_t pixel_base_idx,
    size_t spectrum_idx,
    size_t local_range,
    sycl::local_accessor<float, 1> &local_mem,
    const Data_access& spectrums_d,
    size_t bands_size
) {
        float sum_pixel_values = 0.0f;
        float sum_reference_values = 0.0f;
        float sum_sqrd_pixel_values = 0.0f;
        float sum_sqrd_reference_values = 0.0f;
        float sum_pixel_by_reference_values = 0.0f;

        #pragma unroll 10
        for (size_t i = 0; i < bands_size; i++) {
            float pixel_value = local_mem[pixel_base_idx + i * local_range];
            float spectrum_value = spectrums_d[spectrum_idx * bands_size + i];

            sum_pixel_values += pixel_value;
            sum_reference_values += spectrum_value;

            sum_sqrd_pixel_values += pixel_value * pixel_value;
            sum_sqrd_reference_values += spectrum_value * spectrum_value;

            sum_pixel_by_reference_values += pixel_value * spectrum_value;
        }

        float numerator = bands_size * sum_pixel_by_reference_values - sum_pixel_values * sum_reference_values;
        float denominator = sycl::sqrt(
            (bands_size * sum_sqrd_pixel_values - sum_pixel_values * sum_pixel_values) *
            (bands_size * sum_sqrd_reference_values - sum_reference_values * sum_reference_values)
        );

        return numerator / denominator;
}

namespace Functors {
    template<typename Data_access>
    struct BaseFunctor {
        Data_access img_d, spectrums_d, results_d;
        size_t n_spectrums, n_lines, n_cols, bands_size;
        size_t coalesced_memory_width = 1;

        BaseFunctor(Data_access img_in, 
                    Data_access spectrums_in,
                    Data_access results_in, 
                    size_t n_spectrums_in, 
                    size_t n_lines_in, 
                    size_t n_cols_in, 
                    size_t bands_size_in, 
                    size_t coalesced_memory_width_in)
                            : 
                    img_d(img_in),
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
        static inline const size_t get_range_local_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, size_t max_ND_size, size_t max_local_mem_size) { return 0; }
        static inline const size_t get_results_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool nd) { return 0; }
        static inline const size_t get_local_mem_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, size_t local_range) { return 0; }
        static inline constexpr bool has_ND() { return true; }
        static inline constexpr float results_initial_value() { return 0.f; }

    };

    template<typename Data_access>
    struct ResultsInitilizer : BaseFunctor<Data_access> {
        Data_access results_d;
        float initial_value;

        ResultsInitilizer(Data_access results_in, float initial_value_in) : results_d(results_in), initial_value(initial_value_in) {}

        static inline constexpr size_t get_n_access_points() { return 1; }
        static inline const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return lines * cols; }
        static inline constexpr bool has_ND() { return false; }

        void operator()(sycl::id<1> i) const { results_d[i] = initial_value; }
    };

    template<typename Data_access>
    struct ImgScaler : BaseFunctor<Data_access>{
        Data_access img_d;
        int scale_factor;

        ImgScaler(Data_access img_in, int reflectance_scale_factor) : scale_factor(reflectance_scale_factor), img_d(img_in) {}

        inline static constexpr size_t get_n_access_points() { return 1; }
        inline static const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return lines * cols * bands;}
        inline static constexpr bool has_ND() { return false; }

        void operator()(sycl::id<1> i) const { img_d[i] /= scale_factor; } 
    };  
    
    template<typename Data_access>
    struct ImgSerializer : BaseFunctor<Data_access> {
        Data_access img_read;
        Data_access img_reordered;
        size_t n_cols, n_lines;
        int interleave; //BSQ == 0, BIL == 1, BIP == 2

        ImgSerializer(Data_access img_read_in, Data_access img_reordered_in, size_t n_cols_in, size_t n_lines_in, int interleave_in) 
        : img_read(img_read_in), 
          img_reordered(img_reordered_in),
          n_cols(n_cols_in),
          n_lines(n_lines_in),
          interleave(interleave_in) {}
        
        inline static constexpr size_t get_n_access_points() { return 2; }
        inline static const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return lines * cols; }
        inline static constexpr bool has_ND() { return false; }

        void operator()(sycl::id<1> id) const {
            size_t new_index;
            if(interleave == 1) {   //BIL
                new_index = id;
            }
            else    //not implemented
                new_index = id;
        
            img_reordered[id] = img_read[id];
        } 
    };

    template<typename Data_access>
    struct Euclidean : BaseFunctor<Data_access>{  //there is no need to calculate the sqrt because we can compare the distances without it, saving computation :)

        Euclidean(Data_access img_in, 
                  Data_access spectrums_in, 
                  Data_access results_in, 
                  size_t n_spectrums_in,
                  size_t n_lines_in,
                  size_t n_cols_in,
                  size_t bands_size_in,
                  size_t coalesced_memory_width_in)
            : BaseFunctor<Data_access>(img_in, spectrums_in, results_in, n_spectrums_in, n_lines_in, n_cols_in, bands_size_in, coalesced_memory_width_in) {}

        inline static size_t get_results_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool nd) { 
            if(nd)
                return lines * cols;
            else
                return lines * cols * 2; 
        }

        inline static size_t get_range_global_size(size_t lines, size_t cols, size_t n_bands, size_t n_spectrums, bool has_local_mem) { 
            if(has_local_mem)
                return lines * cols; 
            else
                return lines * cols * n_spectrums;
        }

        inline static size_t get_range_local_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, size_t max_ND_size, size_t max_local_mem_size) { 
            return cols; 
        }

        inline static size_t get_local_mem_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, size_t local_range) {
            //      image portion        spectrum    results portion
            return (local_range * bands)    +    bands    +    local_range; 
        }
          
        static inline constexpr float results_initial_value() { return FLOAT_MAX; }

        //kernel for basic
        void operator()(sycl::id<1> id) const {
            
            size_t wi_id = id.get(0);
            
            size_t img_2D_size = this->n_lines * this->n_cols;
            
            //spectrum to be compared
            size_t spectrum_offset = (wi_id % this->n_spectrums) * this->bands_size;

            size_t img_offset = (wi_id / this->n_spectrums) * this->bands_size;

            //pixel to be compared
            size_t pixel_offset = img_offset / this->n_spectrums;
            
            float sum = 0.f, diff;
            for(int i = 0; i < this->bands_size; i++) {
                diff = this->img_d[img_offset + i] - this->spectrums_d[spectrum_offset + i];
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
        
        //kernel for nd without local mem
        void operator()(sycl::nd_item<1> id) const {                
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();
            size_t group_range = id.get_local_range()[0];

            size_t result_pixel_index = (group_id * group_range) + local_id;

            //bil img offset              line                          line size                     group 1st sample
            size_t img_offset = group_id * (group_range * this->bands_size);

            size_t coalesced_read_start = img_offset + local_id;
            size_t coalesced_read_end = img_offset + (group_range * this->bands_size);
            size_t local_mem_store_index = local_id * this->bands_size;

            id.barrier(sycl::access::fence_space::local_space); //wait until all the data is loaded into local memory

            float sum, diff;
            size_t lowest_index = 0;
            size_t bands_index_start = local_id * this->bands_size;
            size_t bands_index_end = (local_id + 1) * this->bands_size;
            size_t spectrum_band_index;

            for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                sum = 0.f;
                spectrum_band_index = spectrum * this->bands_size;

                for(size_t bands_index = bands_index_start; bands_index < bands_index_end; bands_index++) {
                    diff = this->img_d[bands_index] - this->spectrums_d[spectrum_band_index++];
                    sum += diff * diff;
                }

                if(this->results_d[result_pixel_index] > sum) {
                    this->results_d[result_pixel_index] = sum;
                    lowest_index = spectrum;
                }
            }

            this->results_d[result_pixel_index] = lowest_index;
        }
        
        //kernel for nd whith local mem
        void operator()(sycl::nd_item<1> id, sycl::local_accessor<float, 1> local_mem) const {
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();
            size_t group_range = id.get_group_range()[0];

            size_t result_pixel_index = (group_id * group_range) + local_id;

            //bil img offset              line                          line size                     group 1st sample
            size_t img_offset = group_id * (group_range * this->bands_size);

            size_t coalesced_read_start = img_offset + local_id;
            size_t coalesced_read_end = img_offset + (group_range * this->bands_size);
            size_t local_mem_store_index = local_id * this->bands_size;

            local_mem[0] = 0;

            for(size_t global_read_index = coalesced_read_start ; global_read_index < coalesced_read_end; global_read_index+= this->bands_size)
                local_mem[local_mem_store_index++] = this->img_d[global_read_index];


            float sum, diff;
            size_t lowest_index = 0;
            size_t bands_index_start = local_id * this->bands_size;
            size_t bands_index_end = (local_id + 1) * this->bands_size;
            size_t spectrum_band_index;

            for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                sum = 0.f;
                spectrum_band_index = spectrum * this->bands_size;

                for(size_t bands_index = bands_index_start; bands_index < bands_index_end; bands_index++) {
                    diff = local_mem[bands_index] - this->spectrums_d[spectrum_band_index++];
                    sum += diff * diff;
                }

                if(this->results_d[result_pixel_index] > sum) {
                    this->results_d[result_pixel_index] = sum;
                    lowest_index = spectrum;
                }
            }

            this->results_d[result_pixel_index] = lowest_index;
        }
    };

    template<typename Data_access>
    struct CCM : BaseFunctor<Data_access>{  
        CCM(Data_access img_in, 
            Data_access spectrums_in, 
            Data_access results_in, 
            size_t n_spectrums_in,
            size_t n_lines_in,
            size_t n_cols_in,
            size_t bands_size_in,
            size_t coalesced_memory_width_in)
            : BaseFunctor<Data_access>(img_in, spectrums_in, results_in, n_spectrums_in, n_lines_in, n_cols_in, bands_size_in, coalesced_memory_width_in) {}

        inline static size_t get_results_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool nd) { 
            if(nd)
                return lines * cols;
            else
                return lines * cols * 2; 
        }

        inline static size_t get_range_global_size(size_t lines, size_t cols, size_t n_bands, size_t n_spectrums, bool has_local_mem) { return lines * cols; }

        static size_t get_range_local_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, size_t max_ND_size, size_t max_local_mem_size) {
            size_t spectra_local_mem_needed = (n_spectrums * bands) * sizeof(float);
            
            size_t local_size = 1;
            size_t global_size = lines * cols;
            for(size_t i = 2; i < max_ND_size; i++) {
                if((spectra_local_mem_needed + (i * bands * sizeof(float))) > max_local_mem_size)
                    break;
                
                if(global_size % i == 0)
                    local_size = i;
            }

            return local_size;
        }

        inline static size_t get_local_mem_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, size_t local_range) {
            return (local_range * bands) + (n_spectrums * bands); 
        }
            
        static inline constexpr float results_initial_value() { return -1.1f; }

        //kernel for basic
        void operator()(sycl::id<1> id) const {
            size_t wi_id = id.get(0);

            size_t img_offset = (wi_id / this->n_spectrums) * this->bands_size;

            size_t spectrum_offset = (wi_id % this->n_spectrums) * this->bands_size;

            float sum_pixel_values = 0, sum_reference_values = 0;
            float sum_sqrd_pixel_values = 0, sum_sqrd_reference_values = 0;
            float sum_pixel_by_reference_values = 0;

            float pixel_value, spectrum_value;

            for(size_t i = 0; i < this->bands_size; i++) {
                pixel_value = this->img_d[img_offset + i];
                spectrum_value = this->spectrums_d[spectrum_offset + i];

                sum_pixel_values += pixel_value;
                sum_reference_values += spectrum_value;

                sum_sqrd_pixel_values += pixel_value * pixel_value;
                sum_sqrd_reference_values += spectrum_value * spectrum_value;

                sum_pixel_by_reference_values += pixel_value * spectrum_value;
            }

            //Pearson correlation coefficient formula
            float numerator = this->bands_size * sum_pixel_by_reference_values - sum_pixel_values * sum_reference_values;
            float denominator = sycl::sqrt((this->bands_size * sum_sqrd_pixel_values - sum_pixel_values * sum_pixel_values) * (this->bands_size * sum_sqrd_reference_values - sum_reference_values * sum_reference_values));

            float correlation = numerator / denominator;
            
            size_t img_2D_size = this->n_lines * this->n_cols;
            size_t pixel_offset = wi_id / this->n_spectrums;
            
            //                                                                                                                                                          where the highest value is stored
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> highest_coefficient(this->results_d[img_2D_size + pixel_offset]);
            float read_coefficient = highest_coefficient.load();
            while(correlation > read_coefficient) {
                if(highest_coefficient.compare_exchange_weak(read_coefficient, correlation, sycl::memory_order::relaxed)) {   //compare only if "read_coefficient" remains as the stored value, if so, change it
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> nearest_spectrum(this->results_d[pixel_offset]);
                    nearest_spectrum.store(spectrum_offset / this->bands_size);
                    break;
                }
                read_coefficient = highest_coefficient.load();
            }
        }

        //kernel for ND with local mem
        void operator()(sycl::nd_item<1> id, sycl::local_accessor<float, 1> local_mem) const {
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();
            size_t local_range = id.get_local_range()[0];

            size_t global_img_index_start = (((group_id * local_range + local_id) / this->n_cols) * (this->n_cols * this->bands_size)) + ((group_id * local_range) % this->n_cols) + local_id;
            size_t global_img_index_end = global_img_index_start + (this->n_cols * this->bands_size);

            //copy img to local memory coalesced
            size_t local_stride = local_id;
            for(size_t copy_img = global_img_index_start; copy_img < global_img_index_end; copy_img += this->n_cols) {
                local_mem[local_stride] = this->img_d[copy_img];
                local_stride += local_range;
            }

            float highest_correlation = -1.1f; //the lowest correlation is -1 so every correlation will be higher
            float best_spectrum_index = this->n_spectrums; //incorrect value

            id.barrier(); //wait until the local memory copy is finished

            for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {

                float correlation = compute_correlation(
                    local_id,
                    spectrum,
                    local_range,
                    local_mem,
                    this->spectrums_d,
                    this->bands_size
                );

                if(correlation > highest_correlation) {
                    highest_correlation = correlation;
                    best_spectrum_index = spectrum;
                }
            } 
            this->results_d[group_id * local_range + local_id] = best_spectrum_index;
        }
    
        //kernel for ND without local mem
        void operator()(sycl::nd_item<1> id) const {
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();

            //bil img offset              line                          line size                     group sample
            size_t img_offset = group_id * this->bands_size;
            size_t spectrum_offset = local_id * this->bands_size;

            float sum_pixel_values = 0, sum_reference_values = 0;
            float sum_sqrd_pixel_values = 0, sum_sqrd_reference_values = 0;
            float sum_pixel_by_reference_values = 0;

            float pixel_value, spectrum_value;

            for(size_t i = 0; i < this->bands_size; i++) {
                pixel_value = this->img_d[img_offset + i];
                spectrum_value = this->spectrums_d[spectrum_offset + i];

                sum_pixel_values += pixel_value;
                sum_reference_values += spectrum_value;

                sum_sqrd_pixel_values += pixel_value * pixel_value;
                sum_sqrd_reference_values += spectrum_value * spectrum_value;

                sum_pixel_by_reference_values += pixel_value * spectrum_value;
            }

            //Pearson correlation coefficient formula
            float numerator = this->bands_size * sum_pixel_by_reference_values - sum_pixel_values * sum_reference_values;
            float denominator = sycl::sqrt((this->bands_size * sum_sqrd_pixel_values - sum_pixel_values * sum_pixel_values) * (this->bands_size * sum_sqrd_reference_values - sum_reference_values * sum_reference_values));

            float correlation = numerator / denominator;
            
            //                                                                                                                                                  where the highest value is stored
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> highest_coefficient(this->results_d[group_id]);
            float read_coefficient = highest_coefficient.load();
            while(correlation > read_coefficient)
                highest_coefficient.compare_exchange_weak(read_coefficient, correlation, sycl::memory_order::relaxed);

            id.barrier();   //the lowest value is already in the results
            
            read_coefficient = this->results_d[group_id];
            if(std::fabs(correlation - read_coefficient) < 0.00001f)   //tolerance for float comparison
                highest_coefficient.store(local_id);    //store the index of the nearest spectrum*/
        }
    };
};

#endif
