#include <Results_writer.hpp>

/* stb_image_write - v1.16 - public domain - http://nothings.org/stb
   writes out PNG/BMP/TGA/JPEG/HDR images to C stdio - Sean Barrett 2010-2015*/
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace std;

int write_jpg(int *nearest_materials_image, size_t width, size_t height, string results_file_name){
    
    const int channels = 3, given_colours = 10; //RGB and given colours
    int colors[20 * channels] = {
        255, 0, 0,      //red
        0, 255, 0,      //green
        0, 0, 255,      //blue
        255, 255, 0,    //yellow
        255, 0, 255,    //violet
        0, 255, 255,    //cyan
        255, 255, 255,  //white
        0, 0, 0,        //black
        128, 128, 128,  //light gray
        25, 25, 25,     //gray
        50, 50, 50,
        75, 75, 75,
        100, 100, 100,
        125, 125, 125,
        150, 150, 150,
        175, 175, 175,
        200, 200, 200,
        225, 225, 225,
        50, 100, 150,
        150, 100, 50
    };

    unsigned char* image = new unsigned char[width * height * channels];
    int calculated_index;
    for (int i = 0; i < width * height; i++){
        calculated_index = (nearest_materials_image[i] * channels) % given_colours; //in case there are more materiales (performance test)
        image[channels * i] = static_cast<unsigned char>(colors[calculated_index]);
        image[(channels * i) + 1] = static_cast<unsigned char>(colors[calculated_index + 1]);
        image[(channels * i) + 2] = static_cast<unsigned char>(colors[calculated_index + 2]);
    }

    if(!filesystem::exists(OUTPUT_FOLDER))
        if(!filesystem::create_directory(OUTPUT_FOLDER)){
            cerr << "Error creating output folder. Aborting..." << endl;
            delete[] image;
            return EXIT_FAILURE;
        }

    if (!stbi_write_jpg(results_file_name.c_str(), width, height, channels, image, JPG_MAX_QUALITY)) {
        cerr << "Error creating jpg. Aborting..." << endl;
        delete[] image;
        return EXIT_FAILURE;
    }

    delete[] image;
    return EXIT_SUCCESS;
}

int write_legend(int *nearest_materials_image, size_t n_pixels, string *materials, size_t file_size, string legend_file_name, string times){
    const int n_colors = 20;
    string colors_name[n_colors] = {
        "Red",
        "Green",
        "Blue",
        "Yellow",
        "Magenta",
        "Cyan",
        "White",
        "Black",
        "Ligth Gray",
        "Gray",
        "50",
        "75",
        "100",
        "125",
        "150",
        "175",
        "200",
        "225",
        "asc",
        "desc",
    };

    ofstream out(legend_file_name);
    if(!out){
        cout << "Error writing legend file. Aborting..." << endl;
        return EXIT_FAILURE;
    }

    int* pixels_per_material = (int*)malloc(file_size * sizeof(int));
    for(size_t i = 0; i < file_size; i++)
        pixels_per_material[i] = 0;
    
    for(size_t i = 0; i < n_pixels; i++) 
        pixels_per_material[nearest_materials_image[i]]++;

    for(size_t i = 0; i < file_size; i++){
        out << i+1 << ": " << materials[i] << "   =>   " << colors_name[i] << "   " << pixels_per_material[i] << "/" << n_pixels << "   " << pixels_per_material[i]*100.0/n_pixels << "% of the total pixels" << endl;
    }

    const int column_width = 18;

    out << endl << "Times in milliseconds:" << endl 
    << setw(column_width) << "Initialization"
    << setw(column_width) << "Copy img" 
    << setw(column_width) << "Scale img" 
    << setw(column_width) << "Copy spectrums" 
    << setw(column_width) << "Kernel executed"
    << setw(column_width) << "Total"
    << endl;

    out << times;

    free(pixels_per_material);

    return EXIT_SUCCESS;
}

exit_code create_results(const char* results_file_name, int *nearest_materials_image, size_t width, size_t height, string *materials, size_t n_materials, string times){
    string results_jpg_file_name = string(results_file_name) + JPG_FILE_EXTENSION;
    string results_legend_file_name = string(results_file_name) + LEGEND_FILE_EXTENSION;

    if(write_jpg(nearest_materials_image, width, height, results_jpg_file_name)){
        cerr << "An unexpected error ocurred writing the jpg results file" << endl;
        return EXIT_FAILURE;
    }

    if(write_legend(nearest_materials_image, width * height, materials, n_materials, results_legend_file_name, times)){
        cerr << "An unexpected error ocurred writing the legend results file" << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}