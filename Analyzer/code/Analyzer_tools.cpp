#include <Analyzer_tools.hpp>
#include <unordered_map>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace Analyzer_tools;

#define PERCENTAGE_FACTOR 100

size_t Analyzer_properties::get_spectrums_size() const noexcept {return n_spectrums * envi_properties.bands; }

const unordered_map<string, Analyzer_algorithms> algorithms_mapper {
    {"EUCLIDEAN", EUCLIDEAN},
    {"CCM", CCM}
};

const unordered_map<string, Analyzer_devices> devices_mapper {
    {"GPU", GPU},
    {"CPU", CPU},
    {"ACCELERATOR", ACCELERATOR},
    {"DEFAULT", DEFAULT}
};

template <typename value>
value map(const string key, unordered_map<string, value> mapper){
    typename unordered_map<string, value>::const_iterator mapped = mapper.find(key);

    if(mapped == mapper.end())
        throw runtime_error("ERROR: Error reading analyzer options. Could not find a valid option for \"" + key + "\"");
    else
        return mapped->second;
}

template<typename T>
void initialize_pointer(T*& ptr, size_t ptr_size, bool set_values = false, float value = FLOAT_MAX) {
    ptr = (T*)malloc(ptr_size * sizeof(T));

    if(set_values)
        for(size_t i = 0; i < ptr_size; i++)
            ptr[i] = value;
}

template<template <typename> typename Functor>
exit_code common_analysis(Analyzer_tools::Analyzer_properties& analyzer_properties, sycl::queue& device_q, Analyzer_variant& img_d, Analyzer_variant& spectrums_d, Analyzer_variant& results_d, Event_opt& kernel_finished) {
    using Static_f = Functor<float*>;
    exit_code return_value;

    size_t results_size = Static_f::get_results_size(analyzer_properties.envi_properties.lines, 
                                                      analyzer_properties.envi_properties.samples, 
                                                      analyzer_properties.envi_properties.bands, 
                                                      analyzer_properties.n_spectrums, 
                                                      analyzer_properties.ND_kernel);

    float* final_results_h = (float*)malloc(results_size * sizeof(float));
    Event_opt copied_event;
    Analyzer_tools::copy_to_device(analyzer_properties.USE_ACCESSORS, device_q, results_d, final_results_h, results_size, &copied_event);

    return_value = Analyzer_tools::launch_kernel<Functor>(device_q, kernel_finished, copied_event, analyzer_properties, array{img_d, spectrums_d, results_d}, 
                                                                         analyzer_properties.n_spectrums,
                                                                         analyzer_properties.envi_properties.lines,
                                                                         analyzer_properties.envi_properties.samples,
                                                                         analyzer_properties.envi_properties.bands,
                                                                         analyzer_properties.coalescent_read_size);

    return return_value;
}

namespace Analyzer_tools {

    Analyzer_properties initialize_analyzer(int argc, char** argv) {
        Analyzer_properties p;
        for(int i = 0; i < argc; i++)
            if (argv[i][0] == '-')
                switch (argv[i][1]) {
                    case 'a':  //algorithm
                        p.algorithm = map(argv[++i], algorithms_mapper);
                        break;

                    case 's': //folder of spectrums
                        p.specrums_folder_path = argv[++i];
                        break;

                    case 'i': //folder with image and .hdr file
                        p.image_hdr_folder_path = argv[++i];
                        break;

                    case 'd': //choose device
                        p.device = map(argv[++i], devices_mapper);
                        break;

                    case 'h': //help
                        cout << help_message << endl;
                        exit(EXIT_SUCCESS);
                    
                    default:
                        break;
                }

        if (!p.image_hdr_folder_path) {
            cerr << "ERROR: No path was given for the image and the .hdr file." << endl;
            return p;
        }
        else if (!p.specrums_folder_path) {
            cerr << "ERROR: No path was given for the spectrums." << endl;
            return p;
        }

        return p;
    }

    size_t count_spectrums(const char* path) {
        size_t contador = 0;

        try {
            for (const auto& entry : filesystem::directory_iterator(path)) {
                if (filesystem::is_regular_file(entry.status()))
                    contador++;
                else if(entry.is_directory() && (entry.path().string() != "." || entry.path().string() != ".."))
                    contador += count_spectrums(entry.path().string().c_str());
            }
        } catch (const filesystem::filesystem_error& e) {
            cerr << "ERROR: Could not open spectrums in folder \"" << path << "\"" << endl;
        }

        return contador;
    }

    exit_code read_spectrums(const char* path, float* spectrums, string* names, ENVI_reader::ENVI_properties& properties, int* spectrums_index){
        for (const auto& entry : filesystem::directory_iterator(path)) {
            if(entry.is_regular_file()) {
                if(ENVI_reader::read_spectrum(entry.path().string(), spectrums + (*spectrums_index * properties.bands), names[*spectrums_index], properties))
                    return EXIT_FAILURE;
                *spectrums_index += 1;
            }
            else if(entry.is_directory() && (entry.path().string() != "." || entry.path().string() != "..")) {
                string new_path = entry.path().string();
                read_spectrums(new_path.c_str(), spectrums, names, properties, spectrums_index);
            }
        }
        return EXIT_SUCCESS;
    }

    string get_filename_by_extension(const char* directory, const char* file_extension){
        try {
            for(const auto& entry : filesystem::directory_iterator(directory))
                if(entry.is_regular_file())
                    if(entry.path().string().find(file_extension) != string::npos){
                        filesystem::path path = entry.path();
                        return path.string();
                    }
        } catch (const filesystem::filesystem_error& e) {
            cerr << "ERROR: Could not find directory \"" << directory << "\"" << endl;
            return "";
        }
        return "";      
    }

    exit_code initialize_SYCL_queue(Analyzer_tools::Analyzer_properties& properties, sycl::queue& q) {
        try {
            switch (properties.device) {
                case Analyzer_tools::CPU:
                    q = sycl::queue(sycl::cpu_selector_v);
                    break;

                case Analyzer_tools::GPU:
                    q = sycl::queue(sycl::gpu_selector_v);
                    break;

                case Analyzer_tools::ACCELERATOR:
                    q = sycl::queue(sycl::accelerator_selector_v);
                    break;

                case Analyzer_tools::DEFAULT:
                default:
                    q = sycl::queue(sycl::default_selector_v);
                    break;
            }
        } catch (const sycl::exception& e) {
            cout << "WARNING: The given selector is not available, default selector will be used" << endl;
            try {
                q = sycl::queue();
            } catch (const sycl::exception& e) {
                cerr << "ERROR: It is not posible to use the default selector. Aborting..." << endl;
                return EXIT_FAILURE;
            }
        }

         const auto& platforms = sycl::platform::get_platforms();
        sycl::device best_device;
        int best_score = -1;

        std::cout << "=== Available devices ===\n";

        for (const auto& platform : platforms) {
            for (const auto& dev : platform.get_devices()) {

                std::string type =
                    dev.is_gpu() ? "GPU" :
                    dev.is_cpu() ? "CPU" :
                    dev.is_accelerator() ? "Accelerator" : "Otro";

                std::cout << "Name: " << dev.get_info<sycl::info::device::name>() << "\n";
                std::cout << "  Type: " << type << "\n";
               
            }
    }


        cout << "Executing SYCL kernels on: " << q.get_device().get_info<sycl::info::device::name>() << endl;
        properties.ND_max_item_work_group_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
        if(properties.ND_max_item_work_group_size > 1)
            properties.ND_kernel = true;
        if(!q.get_device().has(sycl::aspect::usm_device_allocations))
            properties.USE_ACCESSORS = true;
        properties.device_local_memory = q.get_device().get_info<sycl::info::device::local_mem_size>();
        std::vector<size_t> sub_group_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
        properties.coalescent_read_size = *max_element(sub_group_sizes.begin(), sub_group_sizes.end());

        return EXIT_SUCCESS;
    }

    exit_code copy_to_device(bool use_accessors, sycl::queue& device_q, Analyzer_variant& ptr_d, float* ptr_h, size_t copy_size, optional<sycl::event>* copied) {

        try {
            if(use_accessors) {
                ptr_d = sycl::buffer<float, 1>(ptr_h, ptr_h + copy_size);
                if(copied)
                    *copied = device_q.submit([](sycl::handler& h) {});     //complete the event with a kernel with nothing 
            }
            else {
                ptr_d = sycl::malloc_device<float>(copy_size, device_q);

                if(copied)
                    *copied = device_q.memcpy(get<float*>(ptr_d), ptr_h, copy_size * sizeof(float));
                else
                    device_q.memcpy(get<float*>(ptr_d), ptr_h, copy_size * sizeof(float));
            }
        } catch (const sycl::exception& e) {
            std::cerr << "ERROR: error when trying to copy to device with SYCL, error message: " << e.what() << std::endl;
            return EXIT_FAILURE;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: error when trying to copy to device, error message: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    exit_code copy_from_device(bool use_accessors, sycl::queue& device_q, float* ptr_h, variant<float*, sycl::buffer<float, 1>>& ptr_d, size_t copy_size, optional<sycl::event>* copied) {
        try {
            if(use_accessors) {
                sycl::host_accessor<float> temp_acc(get<sycl::buffer<float, 1>>(ptr_d));
                std::copy(temp_acc.begin(), temp_acc.begin() + copy_size, ptr_h);
                if(copied)
                    *copied = device_q.submit([](sycl::handler& h) {});     //complete the event with a kernel with nothing 
            }
            else {
                if(copied)
                    *copied = device_q.memcpy(ptr_h, get<float*>(ptr_d), copy_size * sizeof(float));
                else
                    device_q.memcpy(ptr_h, get<float*>(ptr_d), copy_size * sizeof(float));
            }
        } catch (const sycl::exception& e) {
            std::cerr << "ERROR: error when trying to copy to host from device with SYCL, error message: " << e.what() << std::endl;
            return EXIT_FAILURE;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: error when trying to copy to host from device, error message: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    exit_code initialize(Analyzer_tools::Analyzer_properties& analyzer_properties, sycl::queue& device_q, int argc, char** argv) {
        cout << "Reading program execution options..." << endl;

        analyzer_properties = Analyzer_tools::initialize_analyzer(argc, argv);
        if(analyzer_properties.image_hdr_folder_path == nullptr || analyzer_properties.specrums_folder_path == nullptr) {
            cerr << "ERROR: Error initializing analyzer. Aborting..." << endl;
            return EXIT_FAILURE;
        }

        if(Analyzer_tools::initialize_SYCL_queue(analyzer_properties, device_q)) {
            cerr << "ERROR: Error initializing SYCL. Aborting..." << endl;
            return EXIT_FAILURE;
        }

        cout << "Program execution options read with no errors." << endl;
        return EXIT_SUCCESS;
    }

    exit_code read_hdr(Analyzer_tools::Analyzer_properties& analyzer_properties) {
        cout << "Reading .hdr ENVI header file..." << endl;
        string hdr_path = Analyzer_tools::get_filename_by_extension(analyzer_properties.image_hdr_folder_path, HDR_FILE_EXTENSION);
        if (hdr_path.empty()){
            cerr << "ERROR: Could not find an .hdr file in the path: " << analyzer_properties.image_hdr_folder_path << ". Aborting..." << endl;
            return EXIT_FAILURE;
        }
        
        if (ENVI_reader::read_hdr(hdr_path, analyzer_properties.envi_properties)){
            cerr << "ERROR: Error reading .hdr file. Aborting..." << endl;
            return EXIT_FAILURE;
        }
        cout << ".hdr ENVI header file read with no errors." << endl;
        return EXIT_SUCCESS;
    }

    exit_code read_hyperspectral(Analyzer_tools::Analyzer_properties& analyzer_properties, float*& img) {
        cout << "Reading hyperespectral image..." << endl;
        img = (float*)malloc(analyzer_properties.envi_properties.get_image_3Dsize() * sizeof(float));

        switch (analyzer_properties.envi_properties.interleave) {
            case ENVI_reader::Interleave::BIL: {
                string img_path = Analyzer_tools::get_filename_by_extension(analyzer_properties.image_hdr_folder_path, IMG_FILE_EXTENSION);
                if (img_path.empty()){
                    cerr << "ERROR: Could not find an .img file in the path: " << analyzer_properties.image_hdr_folder_path << ". Aborting..." << endl;
                    return EXIT_FAILURE;
                }
                if(ENVI_reader::read_img(img, analyzer_properties.envi_properties, img_path)) {
                    cerr << "ERROR reading hyperespectral image. Aborting..." << endl;
                    free(img);
                    return EXIT_FAILURE;
                }
                break;
            }
            default:
                cerr << "ERROR: The interleave indicated in the .hdr file is not supported. Aborting..." << endl;
                free(img);
                return EXIT_FAILURE;
        }
        cout << "Hyperespectral image read with no errors." << endl;
        return EXIT_SUCCESS;
    }

    exit_code read_spectrums(Analyzer_tools::Analyzer_properties& analyzer_properties, size_t& n_spectrums, float*& spectrums, string*& names) { 
        cout << "Reading sprectrums files..." << endl;
        n_spectrums = Analyzer_tools::count_spectrums(analyzer_properties.specrums_folder_path);
        if(!n_spectrums) {
            cerr << "ERROR: Error counting spectrums" << endl;
            return EXIT_FAILURE;
        }
        spectrums = (float*)malloc(analyzer_properties.envi_properties.bands * n_spectrums * sizeof(float));
        names = new string[n_spectrums];
        int spectrum_index = 0;

        if(Analyzer_tools::read_spectrums(analyzer_properties.specrums_folder_path, spectrums, names, analyzer_properties.envi_properties, &spectrum_index)) {
            cerr << "ERROR: reading spectrums. Aborting..." << endl;
            free(spectrums);
            delete[] names;
            return EXIT_FAILURE;
        }
        cout << "Spectrums files read with no errors." << endl;
        return EXIT_SUCCESS;
    }

    exit_code scale_image(sycl::queue& device_q, Analyzer_tools::Analyzer_properties& analyzer_properties, Analyzer_variant& img_d, Event_opt& img_scaled, bool serialize) {
        cout << "Scaling image by reflectance scale factor..." << endl;
        exit_code ret;
        if(analyzer_properties.envi_properties.reflectance_scale_factor/PERCENTAGE_FACTOR == 1.f) {
            img_scaled = nullopt;
            cout << "Image scaling not needed." << endl;
            return EXIT_SUCCESS;
        }
        try {
            if(serialize) {
                size_t img_size = analyzer_properties.envi_properties.get_image_3Dsize();
                float* img_reordered_ptr = (float*)malloc(img_size * sizeof(float));
                Analyzer_variant img_reordered;

                if(analyzer_properties.USE_ACCESSORS)
                    img_reordered = sycl::buffer<float, 1>(img_reordered_ptr, img_reordered_ptr + img_size);
                else
                    img_reordered = sycl::malloc_device<float>(img_size, device_q);

                ret = Analyzer_tools::launch_kernel<Functors::ImgSerializer>(device_q, img_scaled, img_scaled, analyzer_properties, array{img_d, img_reordered},
                                                                                    analyzer_properties.envi_properties.samples,
                                                                                    analyzer_properties.envi_properties.lines,
                                                                                    analyzer_properties.envi_properties.interleave);

                img_d = img_reordered;
            }
            else
                ret = Analyzer_tools::launch_kernel<Functors::ImgScaler>(device_q, img_scaled, img_scaled, analyzer_properties, array{img_d}, 
                                                                         analyzer_properties.envi_properties.reflectance_scale_factor/PERCENTAGE_FACTOR,
                                                                         analyzer_properties.envi_properties.samples,
                                                                         analyzer_properties.envi_properties.bands);
        } catch (const sycl::exception &e) {
            std::cerr << "Error when launching SYCL kernel to scale image, error message: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        cout << "Image scaled with no errors." << endl;
        return ret;
    }

    exit_code launch_analysis(Analyzer_tools::Analyzer_properties& analyzer_properties, sycl::queue& device_q, Analyzer_variant& img_d, Analyzer_variant& spectrums_d, Analyzer_variant& results_d, Event_opt& kernel_finished) {
        switch (analyzer_properties.algorithm){
            case Analyzer_tools::Analyzer_algorithms::EUCLIDEAN: { return common_analysis<Functors::Euclidean>(analyzer_properties, device_q, img_d, spectrums_d, results_d, kernel_finished); }                
            case Analyzer_tools::Analyzer_algorithms::CCM: { return common_analysis<Functors::CCM>(analyzer_properties, device_q, img_d, spectrums_d, results_d, kernel_finished);}
            
            default:
                cerr << "Error: Algorithm not implemented" << endl;
                return EXIT_FAILURE;
        }
    }    
}