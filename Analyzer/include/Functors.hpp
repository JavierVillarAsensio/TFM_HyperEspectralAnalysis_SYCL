#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <sycl/sycl.hpp>
#include <optional>

#define FLOAT_MAX 3.4028235e+38

SYCL_EXTERNAL float compute_correlation(
    size_t spectrum_idx,
    sycl::local_accessor<float, 1> &local_mem,
    float* pixel,
    size_t bands_size,
    float sum_pixel_values,
    float sum_sqrd_pixel_values,
    float denominator_part
);

namespace Functors {
    template<typename Data_access>
    struct BaseFunctor {
        Data_access img_d, spectrums_d, results_d;
        size_t n_spectrums, n_lines, n_cols, bands_size;
        size_t coalesced_memory_width = 1, reflectance_scale_factor;

        BaseFunctor(Data_access img_in, 
                    Data_access spectrums_in,
                    Data_access results_in, 
                    size_t n_spectrums_in, 
                    size_t n_lines_in, 
                    size_t n_cols_in, 
                    size_t bands_size_in, 
                    size_t coalesced_memory_width_in,
                    size_t reflectance_scale_factor_in)
                            : 
                    img_d(img_in),
                    spectrums_d(spectrums_in),
                    results_d(results_in),
                    n_spectrums(n_spectrums_in),
                    n_lines(n_lines_in),
                    n_cols(n_cols_in),
                    bands_size(bands_size_in),
                    coalesced_memory_width(coalesced_memory_width_in),
                    reflectance_scale_factor(reflectance_scale_factor_in) {}

        BaseFunctor() = default;

        static inline constexpr size_t get_n_access_points() { return 3; }
        static inline const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return cols * lines; }
        static inline const size_t get_range_local_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, size_t max_ND_size, size_t max_local_mem_size, size_t global_size) {           
            size_t local_size = 1;

            for(size_t i = max_ND_size; i > 1; i--) {                    
                if(global_size % i == 0) {
                    local_size = i;
                    break;
                }
            }

            return local_size; 
        }
        static inline const size_t get_results_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool nd) { return lines * cols; }
        static inline const size_t get_local_mem_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, size_t local_range) { return n_spectrums * bands; }
        static inline constexpr bool has_ND() { return true; }
        static inline constexpr bool uses_local_mem() { return true; }

    };

    template<typename Data_access>
    struct ResultsInitilizer : BaseFunctor<Data_access> {
        Data_access results_d;
        float initial_value;

        ResultsInitilizer(Data_access results_in, float initial_value_in) : results_d(results_in), initial_value(initial_value_in) {}

        static inline constexpr size_t get_n_access_points() { return 1; }
        static inline constexpr bool has_ND() { return false; }
        static inline constexpr bool uses_local_mem() { return false; }

        //kernel for basic
        void operator()(sycl::id<1> i) const { results_d[i] = initial_value; }

        //kernel for nd without local mem
        void operator()(sycl::nd_item<1> i) const { results_d[i.get_global_linear_id()] = initial_value; }
    };

    template<typename Data_access>
    struct ImgScaler : BaseFunctor<Data_access>{
        Data_access img_d;
        int scale_factor;
        size_t n_cols;
        size_t bands_size;

        ImgScaler(Data_access img_in, int reflectance_scale_factor, size_t n_cols_in, size_t bands_size_in) 
            : scale_factor(reflectance_scale_factor), img_d(img_in), n_cols(n_cols_in), bands_size(bands_size_in) {}

        inline static constexpr size_t get_n_access_points() { return 1; }
        static inline const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return cols * lines; }
        static inline constexpr bool uses_local_mem() { return false; }

        //kernel for basic
       void operator()(sycl::id<1> i) const { 
            size_t img_offset = i * bands_size;
                #pragma unroll 10
                for(size_t b = 0; b < bands_size; b++)
                    img_d[img_offset++] /= scale_factor; 
        }

        //kernel for nd without local mem
        void operator()(sycl::nd_item<1> i) const { 
            size_t gl_id = i.get_global_linear_id();
            size_t img_offset = ((gl_id / n_cols) * (bands_size * n_cols)) + (gl_id % n_cols);

            #pragma unroll 10
            for(size_t b = 0; b < bands_size; b++)        
                img_d[img_offset + (b * n_cols)] /= scale_factor; 
        }
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
                  size_t coalesced_memory_width_in,
                  size_t reflectance_scale_factor_in)
                : BaseFunctor<Data_access>(
                         img_in, 
                         spectrums_in, 
                         results_in, 
                         n_spectrums_in, 
                         n_lines_in, 
                         n_cols_in, 
                         bands_size_in, 
                         coalesced_memory_width_in, 
                         reflectance_scale_factor_in) {}
          

        //kernel for basic
        void operator()(sycl::id<1> id) const {
            size_t wi_id = id.get(0);
            
            float nearest_distance = FLOAT_MAX;
            float diff;
            size_t lowest_index = 0;

            //bil img offset              line                        line size                  pixel in the line
            size_t img_offset = ((wi_id / this->n_cols) * (this->bands_size * this->n_cols)) + (wi_id % this->n_cols);
            size_t spectrum_band_index = 0;

            for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                float sum = 0.f;

                for(size_t bands_index = 0; bands_index < this->bands_size; bands_index++) {
                    diff = this->img_d[img_offset + (bands_index * this->n_cols)] - this->spectrums_d[spectrum_band_index++];
                    sum += diff * diff;
                }

                if(nearest_distance > sum) {
                    nearest_distance = sum;
                    lowest_index = spectrum;
                }
            }

            this->results_d[wi_id] = lowest_index;
        }
        
        //kernel for nd without local mem
        void operator()(sycl::nd_item<1> id) const {                
            size_t wi_id = id.get_global_linear_id();
            
            float nearest_distance = FLOAT_MAX;
            float diff;
            size_t lowest_index = 0;

            //bil img offset              line                        line size                  pixel in the line
            size_t img_offset = ((wi_id / this->n_cols) * (this->bands_size * this->n_cols)) + (wi_id % this->n_cols);
            size_t spectrum_band_index = 0;

            for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                float sum = 0.f;

                for(size_t bands_index = 0; bands_index < this->bands_size; bands_index++) {
                    diff = this->img_d[img_offset + (bands_index * this->n_cols)] - this->spectrums_d[spectrum_band_index++];
                    sum += diff * diff;
                }

                if(nearest_distance > sum) {
                    nearest_distance = sum;
                    lowest_index = spectrum;
                }
            }

            this->results_d[wi_id] = lowest_index;
        }
        
        //kernel for nd whith local mem
        void operator()(sycl::nd_item<1> id, sycl::local_accessor<float, 1> local_mem) const {
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();
            size_t local_range = id.get_local_range()[0];

            size_t global_img_index_start = (((group_id * local_range + local_id) / this->n_cols) * (this->n_cols * this->bands_size)) + ((group_id * local_range) % this->n_cols) + (local_id % this->n_cols);
            size_t global_img_index_end = global_img_index_start + (this->n_cols * this->bands_size);

            //copy img to local memory coalesced
            size_t local_stride = local_id;
            for(size_t copy_img = global_img_index_start; copy_img < global_img_index_end; copy_img += this->n_cols) {
                local_mem[local_stride] = this->img_d[copy_img];
                local_stride += local_range;
            }

            float sum, diff, nearest_distance = FLOAT_MAX;
            size_t lowest_index = this->n_spectrums;  //wrong value
            size_t spectrum_band_index = 0;

            for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                sum = 0.f;

                for(size_t bands_index = 0; bands_index < this->bands_size; bands_index++) {
                    diff = local_mem[bands_index * local_range + local_id] - this->spectrums_d[spectrum_band_index++];
                    sum += diff * diff;
                }

                if(nearest_distance > sum) {
                    nearest_distance = sum;
                    lowest_index = spectrum;
                }
            }

            this->results_d[(group_id * local_range) + local_id] = lowest_index;
        }
    };

    template<typename Data_access>
    struct CCM : BaseFunctor<Data_access>{  
        static const size_t pixels_per_thread = 50; //to reduce the impact of memory latency
        CCM(Data_access img_in, 
            Data_access spectrums_in, 
            Data_access results_in, 
            size_t n_spectrums_in,
            size_t n_lines_in,
            size_t n_cols_in,
            size_t bands_size_in,
            size_t coalesced_memory_width_in,
            size_t reflectance_scale_factor_in)
            : BaseFunctor<Data_access>(
                   img_in, 
                   spectrums_in, 
                   results_in, 
                   n_spectrums_in, 
                   n_lines_in, 
                   n_cols_in, 
                   bands_size_in, 
                   coalesced_memory_width_in, 
                   reflectance_scale_factor_in) {}

        static inline const size_t get_range_global_size(size_t lines, size_t cols, size_t bands, size_t n_spectrums, bool has_local_mem) { return cols * lines /  pixels_per_thread; }

        //kernel for basic
        void operator()(sycl::id<1> id) const {
            size_t wi_id = id.get(0);

            size_t img_offset = (wi_id / this->n_cols) * (this->bands_size * this->n_cols) + (wi_id % this->n_cols);
            size_t spectrum_offset = 0;

            float sum_pixel_values, sum_reference_values;
            float sum_sqrd_pixel_values, sum_sqrd_reference_values;
            float sum_pixel_by_reference_values;

            float pixel_value, spectrum_value;
            float highest_correlation = -1.1f; //the lowest correlation is -1 so every correlation will be higher
            size_t best_spectrum_index = this->n_spectrums; //incorrect value

            for(size_t p = 0; p < this->pixels_per_thread; p++){
                size_t pixel_offset_2D = (wi_id * pixels_per_thread) + p;
                size_t img_offset = pixel_offset_2D / this->n_cols * (this->n_cols * this->bands_size) + pixel_offset_2D % this->n_cols;

                float highest_correlation = -1.1f; //the lowest correlation is -1 so every correlation will be higher
                float best_spectrum_index = this->n_spectrums; //incorrect value

                size_t spectrum_index = 0;
                if(img_offset >= (this->n_lines * this->n_cols * this->bands_size))
                    break;


                for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {

                    sum_pixel_values = 0;
                    sum_reference_values = 0;
                    sum_sqrd_pixel_values = 0;
                    sum_sqrd_reference_values = 0;
                    sum_pixel_by_reference_values = 0;

                    for(size_t i = 0; i < this->bands_size; i++) {
                        pixel_value = this->img_d[img_offset + (i * this->bands_size)];
                        spectrum_value = this->spectrums_d[spectrum_offset++];

                        sum_pixel_values += pixel_value;
                        sum_reference_values += spectrum_value;

                        sum_sqrd_pixel_values += pixel_value * pixel_value;
                        sum_sqrd_reference_values += spectrum_value * spectrum_value;

                        sum_pixel_by_reference_values += pixel_value * spectrum_value;
                    }

                    //Pearson correlation coefficient formula
                    float numerator = this->bands_size * sum_pixel_by_reference_values - sum_pixel_values * sum_reference_values;
                    float denominator = sycl::sqrt((this->bands_size * sum_sqrd_pixel_values - sum_pixel_values * sum_pixel_values) * (this->bands_size * sum_sqrd_reference_values - sum_reference_values * sum_reference_values));

                    if((numerator / denominator) > highest_correlation) {
                        highest_correlation = (numerator / denominator);
                        best_spectrum_index = spectrum;
                    }
                }
                this->results_d[pixel_offset_2D] = 1;
            }
        }

        //kernel for ND without local mem
        void operator()(sycl::nd_item<1> id) const {
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();
            size_t local_range = id.get_local_range()[0];

            float sum_pixel_values, sum_reference_values;
            float sum_sqrd_pixel_values, sum_sqrd_reference_values;
            float sum_pixel_by_reference_values;

            float pixel_value, spectrum_value;
            float highest_correlation = -1.1f; //the lowest correlation is -1 so every correlation will be higher
            size_t best_spectrum_index = this->n_spectrums; //incorrect value

            for(size_t p = 0; p < this->pixels_per_thread; p++){
                size_t pixel_offset_2D = (group_id * local_range * pixels_per_thread) + (local_range * p) + local_id;
                size_t img_offset = pixel_offset_2D / this->n_cols * (this->n_cols * this->bands_size) + pixel_offset_2D % this->n_cols;

                float highest_correlation = -1.1f; //the lowest correlation is -1 so every correlation will be higher
                float best_spectrum_index = this->n_spectrums; //incorrect value

                size_t spectrum_index = 0;
                if(img_offset >= (this->n_lines * this->n_cols * this->bands_size))
                    break;


                for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {

                    sum_pixel_values = 0;
                    sum_reference_values = 0;
                    sum_sqrd_pixel_values = 0;
                    sum_sqrd_reference_values = 0;
                    sum_pixel_by_reference_values = 0;

                    for(size_t i = 0; i < this->bands_size; i++) {
                        pixel_value = this->img_d[img_offset + (i * this->bands_size)];
                        spectrum_value = this->spectrums_d[spectrum_index++];

                        sum_pixel_values += pixel_value;
                        sum_reference_values += spectrum_value;

                        sum_sqrd_pixel_values += pixel_value * pixel_value;
                        sum_sqrd_reference_values += spectrum_value * spectrum_value;

                        sum_pixel_by_reference_values += pixel_value * spectrum_value;
                    }

                    //Pearson correlation coefficient formula
                    float numerator = this->bands_size * sum_pixel_by_reference_values - sum_pixel_values * sum_reference_values;
                    float denominator = sycl::sqrt((this->bands_size * sum_sqrd_pixel_values - sum_pixel_values * sum_pixel_values) * (this->bands_size * sum_sqrd_reference_values - sum_reference_values * sum_reference_values));

                    if((numerator / denominator) > highest_correlation) {
                        highest_correlation = (numerator / denominator);
                        best_spectrum_index = spectrum;
                    }
                }
                this->results_d[pixel_offset_2D] = img_offset;
            }
        }

        //kernel for ND with local mem
        void operator()(sycl::nd_item<1> id, sycl::local_accessor<float, 1> local_mem) const {
            size_t group_id = id.get_group_linear_id();
            size_t local_id = id.get_local_linear_id();
            size_t local_range = id.get_local_range()[0];

            //copy specs to local memory coalesced
            size_t local_stride = local_id;
            for(size_t copy_specs = local_id; copy_specs < (this->n_spectrums * this->bands_size); copy_specs += local_range) {
                local_mem[local_stride] = this->spectrums_d[copy_specs];
                local_stride += local_range;
            }

            id.barrier(); //wait until the local memory copy is finished
         
            for(size_t p = 0; p < this->pixels_per_thread; p++){
                size_t pixel_offset_2D = (group_id * local_range * pixels_per_thread) + (local_range * p) + local_id;
                size_t img_offset = pixel_offset_2D / this->n_cols * (this->n_cols * this->bands_size) + pixel_offset_2D % this->n_cols;

                float highest_correlation = -1.1f; //the lowest correlation is -1 so every correlation will be higher
                float best_spectrum_index = this->n_spectrums; //incorrect value
                float calculated_correlation;

                size_t spectrum_index = 0;
                if(img_offset >= (this->n_lines * this->n_cols * this->bands_size))
                    break;

                float pixel[256];
                float sum_pixel_values = 0.0f;
                float sum_sqrd_pixel_values = 0.0f;
                float denominator_part;
                for(short i = 0; i < this->bands_size; i++){
                    pixel[i] = this->img_d[img_offset + (i * this->bands_size)]/this->reflectance_scale_factor;
                    sum_pixel_values += pixel[i];
                    sum_sqrd_pixel_values += pixel[i] * pixel[i];
                }
                denominator_part = (this->bands_size * sum_sqrd_pixel_values - sum_pixel_values * sum_pixel_values);

                #pragma unroll 1
                for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                    calculated_correlation = compute_correlation(
                        spectrum * this->bands_size,
                        local_mem,
                        pixel,
                        this->bands_size,
                        sum_pixel_values,
                        sum_sqrd_pixel_values,
                        denominator_part);

                    if(calculated_correlation > highest_correlation) {
                        highest_correlation = calculated_correlation;
                        best_spectrum_index = spectrum;            
                    } 
                }
                this->results_d[pixel_offset_2D] = best_spectrum_index;
            }

            if(group_id == id.get_group_range()[0] - 1){ //last work-group
                size_t pixel_offset_2D_last_pixels = ((group_id + 1) * local_range * pixels_per_thread) + local_id;
                size_t last_pixel_index = this->n_lines * this->n_cols;
                if(pixel_offset_2D_last_pixels < last_pixel_index) {
                    size_t img_offset = (pixel_offset_2D_last_pixels / this->n_cols) * (this->n_cols * this->bands_size) + (pixel_offset_2D_last_pixels % this->n_cols);

                    float highest_correlation = -1.1f; //the lowest correlation is -1 so every correlation will be higher
                    float best_spectrum_index = this->n_spectrums; //incorrect value

                    size_t spectrum_index = 0;

                    for(size_t spectrum = 0; spectrum < this->n_spectrums; spectrum++) {
                        float sum_pixel_values = 0.0f;
                        float sum_reference_values = 0.0f;
                        float sum_sqrd_pixel_values = 0.0f;
                        float sum_sqrd_reference_values = 0.0f;
                        float sum_pixel_by_reference_values = 0.0f;

                        #pragma unroll 10
                        for (size_t i = 0; i < this->bands_size; i++) {
                            size_t specs_index = spectrum_index++;
                            size_t img_index = img_offset + (i * this->n_cols);

                            float pixel_value = this->img_d[img_index]/this->reflectance_scale_factor;
                            float spectrum_value = local_mem[specs_index];

                            sum_pixel_values += pixel_value;
                            sum_reference_values += spectrum_value;

                            sum_sqrd_pixel_values += pixel_value * pixel_value;
                            sum_sqrd_reference_values += spectrum_value * spectrum_value;

                            sum_pixel_by_reference_values += pixel_value * spectrum_value;
                        }

                        float numerator = this->bands_size * sum_pixel_by_reference_values - sum_pixel_values * sum_reference_values;
                        float denominator = sycl::sqrt(
                            (this->bands_size * sum_sqrd_pixel_values - sum_pixel_values * sum_pixel_values) *
                            (this->bands_size * sum_sqrd_reference_values - sum_reference_values * sum_reference_values)
                        );
                        float correlation = numerator / denominator;

                        if(correlation > highest_correlation) {
                            highest_correlation = correlation;
                            best_spectrum_index = spectrum;
                        }
                    } 
                    this->results_d[pixel_offset_2D_last_pixels] = best_spectrum_index;
                }
            }
        }
    };
};

#endif
