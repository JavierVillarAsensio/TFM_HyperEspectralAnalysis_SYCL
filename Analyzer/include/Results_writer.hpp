#ifndef RESULTS_WRITER_H
#define RESULTS_WRITER_H

#include <unistd.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <fstream>

#include <stb_image_write.h>

#define JPG_MAX_QUALITY 100

int write_jpg(int *nearest_materials_image, size_t width, size_t height, const char* result_file_name);
int write_legend(string *materials, size_t file_size, const char* legend_file_name);


#endif