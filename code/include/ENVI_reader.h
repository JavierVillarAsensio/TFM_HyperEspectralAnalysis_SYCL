#ifndef ENVI_READER_H
#define ENVI_READER_H

#include <optional>

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
    };

    std::optional<const ENVI_properties> read_hdr(const char* filename);
    exit_code read_img_bil(float *img, const ENVI_properties properties, const char* filename);
    exit_code check_properties(const ENVI_properties &properties);
}

#endif