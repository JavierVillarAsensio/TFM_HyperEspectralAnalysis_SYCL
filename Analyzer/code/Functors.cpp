#include <Functors.hpp>

SYCL_EXTERNAL float compute_correlation(
    size_t spectrum_idx,
    sycl::local_accessor<float, 1> &local_mem,
    float* pixel,
    size_t bands_size,
    float sum_pixel_values,
    float sum_sqrd_pixel_values,
    float denominator_part
) {
        float sum_reference_values = 0.0f;
        float sum_sqrd_reference_values = 0.0f;
        float sum_pixel_by_reference_values = 0.0f;

        uint16_t specs_index_start = spectrum_idx;
        uint16_t specs_index_end = specs_index_start + bands_size;
        short pixel_idx = 0;

        #pragma unroll 1
        for (short i = specs_index_start; i < specs_index_end; i++) {
            float spectrum_value = local_mem[i];

            sum_reference_values += spectrum_value;
            sum_sqrd_reference_values += spectrum_value * spectrum_value;
            sum_pixel_by_reference_values += pixel[pixel_idx++] * spectrum_value;
        }

        float numerator = bands_size * sum_pixel_by_reference_values - sum_pixel_values * sum_reference_values;
        float denominator = sycl::sqrt(denominator_part * (bands_size * sum_sqrd_reference_values - sum_reference_values * sum_reference_values));

        return numerator / denominator;
}