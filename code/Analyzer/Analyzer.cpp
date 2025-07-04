#include <Analyzer_tools.h>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace std;

template<typename... Args>
void free_malloc_resources(Args*... args) { (free(args), ...); }
template<typename... Args>
void free_new_resources(Args*... args) { (delete[] args, ...); }


int main(int argc, char* argv[]) {
    ////////////////////////////////Initialize////////////////////////////////
    cout << "Reading program execution options..." << endl;
    Analyzer_tools::Analyzer_properties analyzer_properties = Analyzer_tools::initialize_analyzer(argc, argv);
    if(analyzer_properties.image_hdr_folder_path == nullptr || analyzer_properties.specrums_folder_path == nullptr) {
        cerr << "Error initializing analyzer. Aborting..." << endl;
        return EXIT_FAILURE;
    }
    cout << "Program execution options read with no errors." << endl;
    ////////////////////////////////Initialize////////////////////////////////


    /////////////////////////////////Read .hdr////////////////////////////////
    cout << "Reading .hdr ENVI header file..." << endl;
    string hdr_path = Analyzer_tools::get_filename_by_extension(analyzer_properties.image_hdr_folder_path, HDR_FILE_EXTENSION);
    if (hdr_path.empty()){
        cerr << "Error. Could not find an .hdr file in the path: " << analyzer_properties.image_hdr_folder_path << ". Aborting..." << endl;
        return EXIT_FAILURE;
    }
    ENVI_reader::ENVI_properties envi_properties;
    if (ENVI_reader::read_hdr(hdr_path, &envi_properties)){
        cerr << "Error reading .hdr file. Aborting..." << endl;
        return EXIT_FAILURE;
    }
    cout << ".hdr ENVI header file read with no errors." << endl;
    /////////////////////////////////Read .hdr////////////////////////////////
    

    ///////////////////////Read hyperespectral image//////////////////////////
    cout << "Reading hyperespectral image..." << endl;
    float* img = (float*)malloc(envi_properties.get_image_size() * sizeof(float));

    switch (envi_properties.interleave) {
        case ENVI_reader::Interleave::BIL: {
            string img_path = Analyzer_tools::get_filename_by_extension(analyzer_properties.image_hdr_folder_path, IMG_FILE_EXTENSION);
            if (img_path.empty()){
                cerr << "Error. Could not find an .img file in the path: " << analyzer_properties.image_hdr_folder_path << ". Aborting..." << endl;
                return EXIT_FAILURE;
            }
            if(ENVI_reader::read_img_bil(img, &envi_properties, img_path)) {
                cerr << "Error reading hyperespectral image. Aborting..." << endl;
                free_malloc_resources(img);
                return EXIT_FAILURE;
            }
            break;
        }
        
        default:
            cerr << "Error. The interleave indicated in the .hdr file is not supported. Aborting..." << endl;
            free_malloc_resources(img);
            return EXIT_FAILURE;
    }
    cout << "Hyperespectral image read with no errors." << endl;
    ///////////////////////Read hyperespectral image//////////////////////////


    

    ////////////////////////////Read spectrums////////////////////////////////
    cout << "Reading sprectrums files..." << endl;
    size_t n_spectrums = Analyzer_tools::count_spectrums(analyzer_properties.specrums_folder_path);
    if(!n_spectrums) {
        cerr << "Error counting spectrums" << endl;
        free_malloc_resources(img);
        return EXIT_FAILURE;
    }
    float *spectrums = (float*)malloc(envi_properties.bands * n_spectrums * sizeof(float));
    string* names = new string[n_spectrums];
    int spectrum_index = 0;

    if(Analyzer_tools::read_spectrums(analyzer_properties.specrums_folder_path, spectrums, names, envi_properties, &spectrum_index)) {
        cerr << "Error reading spectrums. Aborting..." << endl;
        free_malloc_resources(img, spectrums);
        free_new_resources(names);
        return EXIT_FAILURE;
    }
    cout << "Spectrums files read with no errors." << endl;
    ////////////////////////////Read spectrums////////////////////////////////

    free_malloc_resources(img, spectrums);
    free_new_resources(names);

    return EXIT_SUCCESS;
}