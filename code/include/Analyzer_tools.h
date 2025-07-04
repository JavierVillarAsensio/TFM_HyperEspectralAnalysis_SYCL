#ifndef ANALYZER_TOOLS_H
#define ANALYZER_TOOLS_H

#include <ENVI_reader.h>
#include <string>

#define HDR_FILE_EXTENSION ".hdr"
#define IMG_FILE_EXTENSION ".img" 


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

namespace Analyzer_tools {

    enum Analyzer_algorithms {
        EUCLIDEAN,
        CCM
    };

    enum Analyzer_devices {
        CPU,
        GPU,
        DEFAULT
    };

    struct Analyzer_properties {
        const char* specrums_folder_path = nullptr;
        const char* image_hdr_folder_path = nullptr;
        Analyzer_algorithms algorithm = EUCLIDEAN;
        Analyzer_devices device = DEFAULT;
    };                           

    Analyzer_properties initialize_analyzer(int argc, char** argv);
    size_t count_spectrums(const char* path);
    exit_code read_spectrums(const char* path, float* spectrums, std::string* names, ENVI_reader::ENVI_properties& properties, int* spectrums_index );
    std::string get_filename_by_extension(const char* directory, const char* file_extension);
}

#endif