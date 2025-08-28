#ifndef RESULTS_WRITER_H
#define RESULTS_WRITER_H

#include <unistd.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <fstream>

#define JPG_MAX_QUALITY 100
#define LEGEND_FILE_EXTENSION "_legend.txt"
#define JPG_FILE_EXTENSION ".jpg"

typedef int exit_code;

const std::filesystem::path OUTPUT_FOLDER = "output";

exit_code create_results(const char* results_file_name, int *nearest_materials_image, size_t width, size_t height, std::string *materials, size_t n_materials, std::string times);

#endif