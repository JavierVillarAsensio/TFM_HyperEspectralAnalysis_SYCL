#ifndef ANALYZER_TOOLS_H
#define ANALYZER_TOOLS_H

#include <ENVI_reader.hpp>
#include <Functors.hpp>

#include <string>
#include <sycl/sycl.hpp>
#include <optional>
#include <utility>
#include <array>
#include <Results_writer.hpp>

#define HDR_FILE_EXTENSION ".hdr"
#define IMG_FILE_EXTENSION ".img"

#define FLOAT_MAX 3.4028235e+38

#define HAS_LOCAL_MEM(x) (x.ND_kernel && x.device_local_memory > 1)

using Analyzer_variant = std::variant<float*, sycl::buffer<float, 1>>;
using Result_variant = std::variant<float*, sycl::host_accessor<float>>;
using Range_variant = std::variant<sycl::range<1>, sycl::nd_range<1>>;
using Event_opt = std::optional<sycl::event>;
using Acc = sycl::accessor<float, 1, sycl::access::mode::read_write>;

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
        size_t n_spectrums;
        size_t device_local_memory = 0;
        size_t coalescent_read_size = 1;
        bool USE_ACCESSORS = false;
        size_t ND_max_item_work_group_size = 1;
        ENVI_reader::ENVI_properties envi_properties;
    };    
    
    namespace detail {
        template<template<typename> typename Functor, typename Data_access_type, typename RangeType>
        struct KernelName{};

        template<template <typename> typename Functor, typename Data_access_type, bool use_local_memory, typename... Args>
        inline Functor<Data_access_type> construct_functor(std::array<Data_access_type, Functor<Data_access_type>::get_n_access_points()>& arr, Analyzer_properties& p, sycl::handler& h, Args... args){
            Local_mem_wrapper local_mem_wrapped;
            if constexpr (use_local_memory) {
                auto local_mem = sycl::local_accessor<float, 1>(Functor<Data_access_type>::get_local_mem_size(p.envi_properties.lines, p.envi_properties.samples, p.envi_properties.bands, p.n_spectrums), h);
                local_mem_wrapped = Local_mem_wrapper(local_mem);
            }
            
            return [&]<size_t... Indexes>(std::index_sequence<Indexes...>) {
                    return Functor<Data_access_type> ( arr[Indexes]..., std::forward<Args>(args)..., local_mem_wrapped);
                }(std::make_index_sequence<Functor<Data_access_type>::get_n_access_points()>{});
        }

        template <template <typename> typename Functor, bool use_local_memory, size_t array_size, typename Data_access_type, typename... Args>
        inline auto make_functor(Data_access_type /*detect type*/, std::array<Data_access_type, array_size>& accesses, Analyzer_properties& p, sycl::handler& h, Args&&... args) {
            using Array_type = std::conditional_t<std::is_same_v<Data_access_type, sycl::buffer<float,1>>, std::array<Acc, array_size>, std::array<float*, array_size>>;
            using Array_data_type = std::conditional_t<std::is_same_v<Data_access_type, sycl::buffer<float, 1>>, Acc, float*>;

            Array_type array_in;
            if constexpr (std::is_same_v<Data_access_type, sycl::buffer<float,1>>) {
                for (size_t i = 0; i < array_size; ++i)
                    array_in[i] = accesses[i].template get_access<sycl::access::mode::read_write>(h);

            }
            else if constexpr (std::is_same_v<Data_access_type, float*>)
                array_in = accesses;

            return construct_functor<Functor, Array_data_type, use_local_memory>(array_in, p, h, std::forward<Args>(args)...);
        }

        template <typename Data_access_type, size_t N_ACCESSES, typename ArrayType>
        std::array<Data_access_type, N_ACCESSES> create_array(ArrayType& arr) {
            return [&]<std::size_t... Indexes>(std::index_sequence<Indexes...>) {
                return std::array<Data_access_type, N_ACCESSES>{ std::get<Data_access_type>(std::forward<ArrayType>(arr)[Indexes])... };
            }(std::make_index_sequence<N_ACCESSES>{});
        }

        template<template <typename> typename Functor, typename Data_access_type, typename Range_type, bool use_local_memory, size_t array_size, typename... Args>
        inline sycl::event __launch_kernel(sycl::queue& device_q, Event_opt& opt_dependency, Analyzer_properties& p, Range_type range, std::array<Data_access_type, array_size>&& accesses, Args... args) {  
            using Static_f = Functor<float*>;
            try {
                return device_q.submit([&](sycl::handler& h) {
                    if(opt_dependency.has_value())
                        h.depends_on(opt_dependency.value());

                    auto f = detail::make_functor<Functor, use_local_memory, array_size, Data_access_type>(accesses[0], accesses, p, h, std::forward<Args>(args)...);
                            
                    h.parallel_for<KernelName<Functor, decltype(f.img_d), Range_type>>(range, f);
                });
            } catch (const sycl::exception &e) {
                std::cerr << "Error when launching SYCL kernel, error message: " << e.what() << std::endl;
            }
            return sycl::event {};
        }
    
        template<template <typename> typename Functor>
        Range_variant create_range_variant(Analyzer_properties& p) {
            using Static_f = Functor<float*>;
            Range_variant var_range;
        
            size_t global_size = Static_f::get_range_global_size(p.envi_properties.lines, p.envi_properties.samples, p.envi_properties.bands, p.n_spectrums, HAS_LOCAL_MEM(p));
            size_t local_size = Static_f::get_range_local_size(p.envi_properties.lines, p.envi_properties.samples, p.envi_properties.bands, p.n_spectrums, HAS_LOCAL_MEM(p));

            var_range = p.ND_kernel && p.ND_max_item_work_group_size > local_size && Static_f::has_ND()
                ? var_range = sycl::nd_range<1> {sycl::range<1> {global_size}, sycl::range<1> {local_size}} 
                : sycl::range<1> {global_size};

            return var_range;
        }
    }

    Analyzer_properties initialize_analyzer(int argc, char** argv);
    size_t count_spectrums(const char* path);
    exit_code read_spectrums(const char* path, float* spectrums, std::string* names, ENVI_reader::ENVI_properties& properties, int* spectrums_index );
    std::string get_filename_by_extension(const char* directory, const char* file_extension);
    exit_code initialize_SYCL_queue(Analyzer_tools::Analyzer_properties& properties, sycl::queue& q);
    exit_code copy_to_device(bool use_accessors, sycl::queue& device_q, Analyzer_variant& ptr_d, float* ptr_h, size_t copy_size, Event_opt* copied = nullptr);
    exit_code copy_from_device(bool use_accessors, sycl::queue& device_q, float* ptr_h, Analyzer_variant& ptr_d, size_t copy_size, Event_opt* copied = nullptr);

    template<template <typename> typename Functor, typename... Args>
    inline sycl::event launch_kernel(sycl::queue& device_q, Event_opt& opt_dependency, Analyzer_properties& p, std::array<Analyzer_variant, Functor<float*>::get_n_access_points()>&& variants, Args... args) {
        using Static_f = Functor<float*>;
        sycl::event return_event;
        Range_variant var_range = detail::create_range_variant<Functor>(p);
        
        try {
            std::visit([&](auto&& range) {
                std::visit([&](auto&& first) {
                    using Data_access_type = std::decay_t<decltype(first)>;
                    using Range_type = std::remove_cv_t<std::remove_reference_t<decltype(range)>>;
                    if(HAS_LOCAL_MEM(p) && std::is_same_v<Range_type, sycl::nd_range<1>>)
                        return_event = detail::__launch_kernel<Functor, Data_access_type, Range_type, true, Static_f::get_n_access_points()>
                                (device_q, 
                                 opt_dependency,
                                 p,
                                 range, 
                                 detail::create_array<Data_access_type, Static_f::get_n_access_points(), std::array<Analyzer_variant, Static_f::get_n_access_points()>>(variants), 
                                 std::forward<Args>(args)...);
                    else
                        return_event = detail::__launch_kernel<Functor, Data_access_type, Range_type, false, Static_f::get_n_access_points()>
                                (device_q, 
                                 opt_dependency,
                                 p,
                                 range, 
                                 detail::create_array<Data_access_type, Static_f::get_n_access_points(), std::array<Analyzer_variant, Static_f::get_n_access_points()>>(variants), 
                                 std::forward<Args>(args)...);

                }, variants[0]);
            }, var_range);
        } catch (const std::exception& e) {
            std::cout << "Error when creating the kernel, error message: " << e.what() << std::endl;
        }
        return return_event;
    }
}

#endif