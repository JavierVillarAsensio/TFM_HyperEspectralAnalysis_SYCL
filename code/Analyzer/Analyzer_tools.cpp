#include <Analyzer_tools.h>
#include <unordered_map>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace Analyzer_tools;

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
                    q = sycl::queue([](const sycl::device& dev) {
                        return dev.is_cpu();
                    });
                    break;

                case Analyzer_tools::GPU:
                    q = sycl::queue([](const sycl::device& dev) {
                        return dev.is_gpu();
                    });
                    break;

                case Analyzer_tools::ACCELERATOR:
                    q = sycl::queue([](const sycl::device& dev) {
                        return dev.is_accelerator();
                    });
                    break;

                case Analyzer_tools::DEFAULT:
                default:
                    q = sycl::queue();
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

        properties.ND_max_item_work_group_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
        if(properties.ND_max_item_work_group_size > 1)
            properties.ND_kernel = true;
        if(q.get_device().has(sycl::aspect::usm_device_allocations))
            properties.USM = true;

        return EXIT_SUCCESS;
    }
}