#ifndef ENVI_READER_H
#define ENVI_READER_H

#include <string>

typedef int exit_code;
constexpr exit_code FAILURE = -1;

namespace ENVI_reader {
    
    enum Interleave {
        BSQ,
        BIL,
        BIP,
        FAILURE = -1
    };

    struct ENVI_properties {
        size_t samples = FAILURE;
        size_t lines = FAILURE;
        size_t bands = FAILURE;
        size_t header_offset = FAILURE;
        size_t data_type_size = FAILURE;
        enum Interleave interleave = FAILURE;
        int wavelength_unit;
        int reflectance_scale_factor = FAILURE;
        float* wavelengths = nullptr;
        size_t get_image_3Dsize() const noexcept;
        size_t get_image_2Dsize() const noexcept;
        ~ENVI_properties() noexcept;
    };

    exit_code read_hdr(const std::string filename, ENVI_reader::ENVI_properties& properties);
    exit_code read_img(float *img, const ENVI_reader::ENVI_properties& properties, const std::string filename);
    exit_code check_properties(const ENVI_reader::ENVI_properties& properties);
    exit_code read_spectrum(const std::string filename, float* reflectances, std::string &name, ENVI_reader::ENVI_properties& properties);
};

#endif