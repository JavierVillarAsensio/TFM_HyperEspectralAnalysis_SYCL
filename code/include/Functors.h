#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <sycl/sycl.hpp>
#include <ENVI_reader.h>

namespace Functors {
    namespace USM {
        struct ImgScaler {
            float* img_d;
            int scale_factor;

            ImgScaler(float* img_in, int reflectance_scale_factor) : img_d(img_in), scale_factor(reflectance_scale_factor) {}

            void operator()(sycl::id<1> i) const { img_d[i] /= scale_factor; }
        }; 
    };
    namespace Accessors {
        struct BaseAccessors {
            sycl::accessor<float, 1, sycl::access_mode::read_write> img_acc;
            sycl::accessor<float, 1, sycl::access_mode::read> spectrums_acc;
            sycl::accessor<float, 1, sycl::access_mode::read_write> results_acc;

            BaseAccessors(
                sycl::accessor<float, 1, sycl::access_mode::read_write> img,
                sycl::accessor<float, 1, sycl::access_mode::read> spectrums,
                sycl::accessor<float, 1, sycl::access_mode::read_write> results)
                : img_acc(img), spectrums_acc(spectrums), results_acc(results) {}
        };

        struct ImgScaler {
            int scale_factor;
            sycl::accessor<float, 1, sycl::access_mode::read_write> img_acc;

            ImgScaler(
                sycl::accessor<float, 1, sycl::access_mode::read_write> img,
                int reflectance_scale_factor)
                : img_acc(img), scale_factor(reflectance_scale_factor) {}

            void operator()(sycl::id<1> i) const { img_acc[i] /= scale_factor; }
        }; 
    };
};

#endif
