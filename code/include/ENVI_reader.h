#ifndef ENVI_READER_H
#define ENVI_READER_H

#include <optional>
#include <string>

typedef int exit_code;
constexpr exit_code FAILURE = -1;

namespace ENVI_reader {
    
    enum Interleave {
        BSQ,
        BIL,
        BIP,
        ERROR
    };

    struct ENVI_properties {
        int samples = FAILURE;
        int lines = FAILURE;
        int bands = FAILURE;
        int header_offset = FAILURE;
        int data_type_size = FAILURE;
        enum Interleave interleave;
        int wavelength_unit;
        float reflectance_scale_factor = FAILURE;
        float* wavelengths = nullptr;
        size_t get_image_size() const;
        ~ENVI_properties();
    };

    std::optional<const ENVI_reader::ENVI_properties> read_hdr(const char* filename);
    exit_code read_img_bil(float *img, const ENVI_reader::ENVI_properties properties, const char* filename);
    exit_code check_properties(const ENVI_reader::ENVI_properties &properties);
    exit_code read_spectrum(std::string filename, float* reflectances, std::string &name, ENVI_reader::ENVI_properties& properties);
}

#endif