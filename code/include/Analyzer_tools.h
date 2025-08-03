#ifndef ANALYZER_TOOLS_H
#define ANALYZER_TOOLS_H

#include <ENVI_reader.h>
#include <Functors.h>
#include <string>
#include <sycl/sycl.hpp>
#include <optional>
#include <tuple>
#include <utility>
#include <array>

#define HDR_FILE_EXTENSION ".hdr"
#define IMG_FILE_EXTENSION ".img"

#define FLOAT_MAX 3.4028235e+38


constexpr const char* help_message = "These are the options to execute the program:\n"
                                        "\t-h Shows this message\n"
                                        "\t-a Choose the algorithm of the analyzer, the availabes are: CCM, EUCLIDEAN (EUCLIDEAN by default)\n"
                                        "\t-s Path to the folder where the spectrums are, a folder with folders is also valid "
                                        "but the regular files contained can only be spectrums, otherwise it will cause an error\n"
                                        "\t-i Path to the folder where the image and the .hdr file are. Only one of each should be"
                                        " found in this folder\n"
                                        "\t-d Choose the device selector: CPU, GPU, ACCELERATOR or DEFAULT (DEFAULT by default)\n"
                                      "If -h option the rest will be ignored and the program will be exited, the rest of the "
                                      "options are compulsory except the ones with default values.";

namespace Analyzer_tools {

    enum Analyzer_algorithms {
        EUCLIDEAN,
        CCM
    };

    enum Analyzer_devices {
        CPU,
        GPU,
        ACCELERATOR,
        DEFAULT
    };

    struct Analyzer_properties {
        const char* specrums_folder_path = nullptr;
        const char* image_hdr_folder_path = nullptr;
        Analyzer_algorithms algorithm = EUCLIDEAN;
        Analyzer_devices device = DEFAULT;
        bool ND_kernel = false;
        size_t device_local_memory = 0;
        size_t coalescent_read_size = 1;
        bool USE_ACCESSORS = false;
        size_t ND_max_item_work_group_size = 1;
    };                           

    Analyzer_properties initialize_analyzer(int argc, char** argv);
    size_t count_spectrums(const char* path);
    exit_code read_spectrums(const char* path, float* spectrums, std::string* names, ENVI_reader::ENVI_properties& properties, int* spectrums_index );
    std::string get_filename_by_extension(const char* directory, const char* file_extension);
    exit_code initialize_SYCL_queue(Analyzer_tools::Analyzer_properties& properties, sycl::queue& q);
    exit_code copy_to_device(bool use_accessors, sycl::queue& device_q, std::variant<float*, sycl::buffer<float, 1>>& ptr_d, float* ptr_h, size_t copy_size, std::optional<sycl::event>* copied = nullptr);
    exit_code copy_from_device(bool use_accessors, sycl::queue& device_q, std::variant<float*, sycl::host_accessor<float>>& ptr_h, std::variant<float*, sycl::buffer<float, 1>>& ptr_d, size_t copy_size, std::optional<sycl::event>* copied = nullptr);

    template<
        template <typename> typename Functor,
        typename Data_access_type,
        size_t N_VARIANTS,
        typename... Args
    >
    Functor<Data_access_type> create_functor(std::array<Data_access_type, N_VARIANTS>& arr, Args&&... args){
        return [&]<size_t... Indexes>(std::index_sequence<Indexes...>) {
            return Functor<Data_access_type> ( arr[Indexes]..., std::forward<Args>(args)... );
        }(std::make_index_sequence<N_VARIANTS>{});
    }

    template <
        typename Data_access_type,
        size_t N_ACCESSES,
        typename ArrayType
    >
    std::array<Data_access_type, N_ACCESSES> create_array(ArrayType& arr) {
        return [&]<std::size_t... Indexes>(std::index_sequence<Indexes...>) {
            return std::array<Data_access_type, N_ACCESSES>{ std::get<Data_access_type>(std::forward<ArrayType>(arr)[Indexes])... };
        }(std::make_index_sequence<N_ACCESSES>{});
    }

    template<
        template <typename> typename Functor,
        typename Range_type,
        size_t N_VARIANTS, 
        typename... Args
    >
    sycl::event launch_kernel(sycl::queue& device_q, 
                                            Range_type range, 
                                            std::optional<sycl::event>& opt_dependency,
                                            std::array<std::variant<float*, sycl::buffer<float, 1>>, N_VARIANTS>&& variants, 
                                            Args&&... args) {

        sycl::event return_event;

        std::visit([&](auto&& first) {
            using Data_access_type = std::decay_t<decltype(first)>;
            std::array<Data_access_type, N_VARIANTS> accesses = create_array<Data_access_type, N_VARIANTS, std::array<std::variant<float*, sycl::buffer<float, 1>>, N_VARIANTS>>(variants);

            try {
                return_event = device_q.submit([&](sycl::handler& h) {
                    if(opt_dependency.has_value())
                        h.depends_on(opt_dependency.value());

                    if constexpr (std::is_same_v<Data_access_type, float*>) {

                        if (N_VARIANTS == 3) {
                            float* img = accesses[0];
                            sycl::ext::oneapi::experimental::printf("img: %f %f %f %f %f %f %f %f %f %f %f %f \n", img[0], img[1], img[2], img[3], img[4], img[5], img[6], img[7], img[8], img[9], img[10], img[11]);

                            float* specs = accesses[1];
                            sycl::ext::oneapi::experimental::printf("spec: %f %f %f %f\n", specs[0], specs[1], specs[2], specs[3]);

                            float* res = accesses[2];
                            sycl::ext::oneapi::experimental::printf("res: %f %f %f %f %f %f\n", res[0], res[1], res[2], res[3], res[4], res[5]);
                        }

                        Functor<float*> f =  create_functor<Functor, Data_access_type, N_VARIANTS>(accesses, std::forward<Args>(args)...);
                        h.parallel_for(range, f);
                    }
                    else if constexpr (std::is_same_v<Data_access_type, sycl::buffer<float, 1>>) {
                        std::array<sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::device>, N_VARIANTS> accessors_arr{};
                        for(size_t access_index = 0; access_index < N_VARIANTS; access_index++)
                            accessors_arr[access_index] = sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::device> {accesses[access_index], h};

                        Functor<sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::device>> f =  
                            create_functor<Functor, sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::device>, N_VARIANTS>(accessors_arr, std::forward<Args>(args)...);
                        h.parallel_for(range, f);
                    }
                });
            } catch (const sycl::exception &e) {
                std::cerr << "Error when launching SYCL kernel, error message: " << e.what() << std::endl;
            }
        }, variants[0]);

        return return_event;
    }
}

#endif