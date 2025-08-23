#include <Results_writer.hpp>

using namespace std;

int write_jpg(int *nearest_materials_image, size_t width, size_t height, const char* results_file_name){
    
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

    if (!stbi_write_jpg(results_file_name, width, height, channels, image, JPG_MAX_QUALITY)) {
        cout << "Error creating jpg. Aborting..." << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int write_legend(string *materials, size_t file_size, const char* legend_file_name){
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

    for(int i = 0; i < file_size; i++){
        out << i+1 << ": " << materials[i] << "   =>   " << colors_name[i] << endl;
    }

    return EXIT_SUCCESS;
}