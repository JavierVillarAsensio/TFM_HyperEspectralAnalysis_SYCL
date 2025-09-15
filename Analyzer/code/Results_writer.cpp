#include <Results_writer.hpp>
#include <matio.h>
#include <unordered_map>

/* stb_image_write - v1.16 - public domain - http://nothings.org/stb
   writes out PNG/BMP/TGA/JPEG/HDR images to C stdio - Sean Barrett 2010-2015*/
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define MAT_FILE "Analyzer/jasperRidge2_R198/end4.mat"
#define CONFUSION_FILE "output/confusion_matrix.txt"
#define COMPARATION_FILE "output/confusion"
#define JR_N_ENDMEMBERS 4
#define FLOAT_MIN 1.17549e-38

using namespace std;

string endmembers[JR_N_ENDMEMBERS] = {"tree", "water", "dirt", "road"};

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

int write_comparison(int *nearest_materials_image, size_t *to_compare, size_t img_size, size_t width, size_t height){
    int *comparation = (int*)malloc(img_size * sizeof(int));
    int coincidences = 0, confusion_matrix[JR_N_ENDMEMBERS][JR_N_ENDMEMBERS], image_nearest, to_compare_pixel;
    for(int i = 0; i < JR_N_ENDMEMBERS; i++)
        for(int j = 0; j < JR_N_ENDMEMBERS; j++)
            confusion_matrix[i][j] = 0;

    for(int pixel = 0; pixel < img_size; pixel++){
        image_nearest = nearest_materials_image[pixel];
        to_compare_pixel = to_compare[pixel];
        confusion_matrix[image_nearest][to_compare_pixel]++;
        if(nearest_materials_image[pixel] == to_compare[pixel]){
            comparation[pixel] = 1;
            coincidences++;
        }
        else
            comparation[pixel] = 0;
    }
    cout << "Comparing results with JasperRidge..." << endl;
    ofstream out(CONFUSION_FILE);
    if(!out){
        cout << "Error writing confusion matrix file. Aborting..." << endl;
        return EXIT_FAILURE;
    }

    for(int i = 0; i < JR_N_ENDMEMBERS; i++){
        for(int j = 0; j < JR_N_ENDMEMBERS; j++)
            out << confusion_matrix[i][j] << " ";
        out << endl;
    }

    const int channels = 3; //RGB
    int colors[2 * channels] = {
        255, 0, 0,      //red
        0, 255, 0     //green
    };

    unsigned char* image = (unsigned char*)malloc(width * height * channels);

    for (int i = 0; i < img_size; i++){
        image[channels * i] = static_cast<unsigned char>(colors[comparation[i] * channels]);
        image[(channels * i) + 1] = static_cast<unsigned char>(colors[(comparation[i] * channels) + 1]);
        image[(channels * i) + 2] = static_cast<unsigned char>(colors[(comparation[i] * channels) + 2]);
    }
    free(comparation);

    float perc_coincidence = ((float)coincidences/(float)img_size)*100;
    string str = to_string(perc_coincidence);
    size_t decimal_pos = str.find('.');
    if (decimal_pos != string::npos) {
        str = str.substr(0, decimal_pos + 3); // +3 porque queremos 2 decimales después del punto
    }
    string filename_str = COMPARATION_FILE + str + ".jpg";
    const char *filename = filename_str.c_str();

    if (!stbi_write_jpg(filename, width, height, channels, image, 100)) {
        cout << "Error creating comparation jpg. Aborting..." << endl;
        free(image);
        return EXIT_FAILURE;
    }
    free(image);

    filesystem::permissions(filename, filesystem::perms::owner_all | filesystem::perms::group_all, filesystem::perm_options::add);
    return EXIT_SUCCESS;
}

exit_code compare_result(int *nearest_materials_image, Analyzer_tools::Analyzer_properties& p, string *materials) {
    mat_t *mat;
    matvar_t *matvar;
    size_t img_size = p.envi_properties.get_image_2Dsize();
    int *aux_to_compare = (int*)malloc(img_size * sizeof(int));
    
    int endmember_index_to_spectrum_index[JR_N_ENDMEMBERS];
    for(int translation = 0; translation < JR_N_ENDMEMBERS; translation++)
        for(int spectrum = 0; spectrum < JR_N_ENDMEMBERS; spectrum++)
            if(materials[spectrum] == endmembers[translation])
                endmember_index_to_spectrum_index[translation] = spectrum;

    mat = Mat_Open(MAT_FILE, MAT_ACC_RDONLY);
    if (mat == nullptr) {
        cerr << "Error opening MAT file" << endl;
        return EXIT_FAILURE;
    }

    matvar_t *matVar = Mat_VarRead(mat, (char*)"A") ;
    if(matVar)
    {
        unsigned xSize = matVar->nbytes / matVar->data_size;
        const double *xData = static_cast<const double*>(matVar->data);

        int n_pixels_to_compare = xSize/JR_N_ENDMEMBERS;
        double abundance, aux_abundance;
        int most_abundant;
        for(int pixel=0; pixel<n_pixels_to_compare; pixel++)
        {
            abundance = FLOAT_MIN;
            most_abundant = -1;
            
            for (int mat_index = 0; mat_index < JR_N_ENDMEMBERS; mat_index++)
            {
                aux_abundance = xData[(pixel*JR_N_ENDMEMBERS) + mat_index];
                if(aux_abundance > abundance){
                    abundance = aux_abundance;
                    most_abundant = mat_index;
                }
            }
            aux_to_compare[pixel] = endmember_index_to_spectrum_index[most_abundant];
        }
    }
    else {
        cerr << "Error reading .mat file" << endl;
        return EXIT_FAILURE;
    }
    Mat_Close(mat);

    size_t *to_compare = (size_t*)malloc(img_size * sizeof(size_t)); 
    size_t width = p.envi_properties.samples, height = p.envi_properties.lines, row, column;
    for (size_t i = 0; i < img_size; i++)
    {
        row = (i+1) % width;
        column = i / height;
        to_compare[i] = aux_to_compare[row * height + column];
    }
    free(aux_to_compare);
    
    if(write_comparison(nearest_materials_image, to_compare, img_size, width, height)) {
        return EXIT_FAILURE;
        free(to_compare);
    }
    free(to_compare);

    return EXIT_SUCCESS;
}