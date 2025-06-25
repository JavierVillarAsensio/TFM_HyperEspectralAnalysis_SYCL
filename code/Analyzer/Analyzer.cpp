#include "ENVI_reader.h"
#include "Analyzer_tools.h"
#include <iostream>

using namespace std;

template<typename... Args>
void free_malloc_resources(Args*... args) { (free(args), ...); }
template<typename... Args>
void free_new_resources(Args*... args) { (delete[] args, ...); }

int main(int argc, char* argv[]) {
    //Initialize
    cout << "Reading program execution options..." << endl;
    Analyzer_properties analyzer_properties = initialize_analyzer(argc, argv);
    cout << "Program execution options read wit no errors." << endl;



    //Read .hdr
    cout << "Reading .hdr ENVI header file..." << endl;
    optional<const ENVI_reader::ENVI_properties> opt_properties = ENVI_reader::read_hdr("a");
    if(!opt_properties) {
        cerr << "Error reading .hdr ENVI header file. Aborting..." << endl;
        return EXIT_FAILURE;
    }
    ENVI_reader::ENVI_properties envi_properties = opt_properties.value();
    cout << ".hdr ENVI header file read with no errors." << endl;



    //Read hyperespectral image
    cout << "Reading hyperespectral image..." << endl;
    float *img = (float*)malloc(envi_properties.get_image_size() * sizeof(float));
    if(ENVI_reader::read_img_bil(img, envi_properties, "a")) {
        cerr << "Error reading hyperespectral image. Aborting..." << endl;
        free_malloc_resources(img);
        return EXIT_FAILURE;
    }
    cout << "Hyperespectral image read with no errors." << endl;



    //Read spectrums
    cout << "Reading sprectrums files..." << endl;
    size_t n_spectrums = count_spectrums(analyzer_properties.specrums_folder_path);
    float *spectrums = (float*)malloc(envi_properties.bands * n_spectrums * sizeof(float));
    string* names = new string[n_spectrums];
    if(read_spectrums(analyzer_properties.specrums_folder_path, spectrums, names, envi_properties)) {
        cerr << "Error reading spectrums. Aborting..." << endl;
        free_malloc_resources(img);
        free_new_resources(names);
        return EXIT_FAILURE;
    }
    cout << "Spectrums files read with no errors." << endl;



    return EXIT_SUCCESS;
}