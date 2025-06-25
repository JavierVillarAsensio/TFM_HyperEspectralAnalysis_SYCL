#ifndef ANALYZER_TOOLS_H
#define ANALYZER_TOOLS_H

#include "ENVI_reader.h"
#include <unordered_map>
#include <string>
#include <iostream>
#include <filesystem>

using namespace std;

constexpr const char* help_message = "These are the options to execute the program:\n"
                               "\t-h Shows this message\n"
                               "\t-a Choose the algorithm of the analyzer, the availabes are: CCM, EUCLIDEAN (EUCLIDEAN by default)\n"
                               "\t-s Path to the folder where the spectrums are, a folder with folders is also valid "
                               "but the regular files contained can only be spectrums, otherwise it will cause an error\n"
                               "\t-i Path to the folder where the image and the .hdr file are. Only one of each should be"
                               " found in this folder\n"
                               "\t-d Choose the device selector: CPU, GPU or DEFAULT (DEFAULT by default)\n"
                               "If -h option the rest will be ignored and the program will be exited, the rest of the "
                               "options are compulsory except the ones with default values.";

struct Analyzer_properties {
    const char* specrums_folder_path = nullptr;
    const char* image_hdr_folder_path = nullptr;
    Analyzer_algorithms algorithm = EUCLIDEAN;
    Analyzer_devices device = DEFAULT;
};                           

enum Analyzer_algorithms {
    EUCLIDEAN,
    CCM
};

enum Analyzer_devices {
    CPU,
    GPU,
    DEFAULT
};

const unordered_map<string, Analyzer_algorithms> algorithms_mapper {
    {"EUCLIDEAN", EUCLIDEAN},
    {"CCM", CCM}
};


const unordered_map<string, Analyzer_devices> devices_mapper {
    {"GPU", GPU},
    {"CPU", CPU},
    {"DEFAULT", DEFAULT}
};

template <typename value>
value map(const string key, unordered_map<string, value> mapper){
    typename unordered_map<string, value>::const_iterator mapped = mapper.find(key);

    if(mapped == mapper.end())
        throw runtime_error("Error reading analyzer options. Could not find a valid option for \"" + key + "\"");
    else
        return mapped->second;
}

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
        cerr << "Error. No path was given for the image and the .hdr file." << endl;
        exit(EXIT_FAILURE);
    }
    else if (!p.specrums_folder_path) {
        cerr << "Error. No path was given for the spectrums." << endl;
        exit(EXIT_FAILURE);
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
        cerr << "Error, could not open spectrums folder" << path << endl;
    }

    return contador;
}

exit_code read_spectrums(const char* path, float* spectrums, string* names, ENVI_reader::ENVI_properties& properties, int spectrums_index = 0){
    for (const auto& entry : filesystem::directory_iterator(path)) {
        if(entry.is_regular_file()) {
            ENVI_reader::read_spectrum(entry.path().string(), spectrums + (spectrums_index * properties.bands), names[spectrums_index], properties);
            spectrums_index++;
        }
        else if(entry.is_directory() && (entry.path().string() != "." || entry.path().string() != ".."))
            read_spectrums(path, spectrums, names, properties, spectrums_index);
    }
}

#endif