#include <Analyzer_tools.h>
#include <iostream>
#include <Functors.h>

using namespace std;

template<typename... Ptrs>
void free_malloc_resources(sycl::event event, Ptrs... ptrs) { event.wait(); ( free(ptrs), ... ); }
template<typename... Ptrs>
void free_USM_resources(sycl::queue& q, Ptrs... ptrs) { (sycl::free(std::get<float*>(ptrs), q), ...); }

exit_code initialize(Analyzer_tools::Analyzer_properties& analyzer_properties, sycl::queue& device_q, int argc, char** argv) {
    cout << "Reading program execution options..." << endl;

    analyzer_properties = Analyzer_tools::initialize_analyzer(argc, argv);
    if(analyzer_properties.image_hdr_folder_path == nullptr || analyzer_properties.specrums_folder_path == nullptr) {
        cerr << "ERROR: Error initializing analyzer. Aborting..." << endl;
        return EXIT_FAILURE;
    }

    if(Analyzer_tools::initialize_SYCL_queue(analyzer_properties, device_q)) {
        cerr << "ERROR: Error initializing SYCL. Aborting..." << endl;
        return EXIT_FAILURE;
    }

    cout << "Program execution options read with no errors." << endl;
    return EXIT_SUCCESS;
}

exit_code read_hdr(Analyzer_tools::Analyzer_properties& analyzer_properties, ENVI_reader::ENVI_properties& envi_properties) {
    cout << "Reading .hdr ENVI header file..." << endl;
    string hdr_path = Analyzer_tools::get_filename_by_extension(analyzer_properties.image_hdr_folder_path, HDR_FILE_EXTENSION);
    if (hdr_path.empty()){
        cerr << "ERROR: Could not find an .hdr file in the path: " << analyzer_properties.image_hdr_folder_path << ". Aborting..." << endl;
        return EXIT_FAILURE;
    }
    
    if (ENVI_reader::read_hdr(hdr_path, &envi_properties)){
        cerr << "ERROR: Error reading .hdr file. Aborting..." << endl;
        return EXIT_FAILURE;
    }
    cout << ".hdr ENVI header file read with no errors." << endl;
    return EXIT_SUCCESS;
}

exit_code read_hyperspectral(Analyzer_tools::Analyzer_properties& analyzer_properties, ENVI_reader::ENVI_properties& envi_properties, float*& img) {
    cout << "Reading hyperespectral image..." << endl;
    img = (float*)malloc(envi_properties.get_image_3Dsize() * sizeof(float));

    switch (envi_properties.interleave) {
        case ENVI_reader::Interleave::BIL: {
            string img_path = Analyzer_tools::get_filename_by_extension(analyzer_properties.image_hdr_folder_path, IMG_FILE_EXTENSION);
            if (img_path.empty()){
                cerr << "ERROR: Could not find an .img file in the path: " << analyzer_properties.image_hdr_folder_path << ". Aborting..." << endl;
                return EXIT_FAILURE;
            }
            if(ENVI_reader::read_img_bil(img, &envi_properties, img_path)) {
                cerr << "ERROR reading hyperespectral image. Aborting..." << endl;
                free(img);
                return EXIT_FAILURE;
            }
            break;
        }
        default:
            cerr << "ERROR: The interleave indicated in the .hdr file is not supported. Aborting..." << endl;
            free(img);
            return EXIT_FAILURE;
    }
    cout << "Hyperespectral image read with no errors." << endl;
    return EXIT_SUCCESS;
}

exit_code read_spectrums(Analyzer_tools::Analyzer_properties& analyzer_properties, ENVI_reader::ENVI_properties& envi_properties, float*& img,  
                         size_t& n_spectrums, float*& spectrums, string*& names) {
    cout << "Reading sprectrums files..." << endl;
    n_spectrums = Analyzer_tools::count_spectrums(analyzer_properties.specrums_folder_path);
    if(!n_spectrums) {
        cerr << "ERROR: Error counting spectrums" << endl;
        free(img);
        return EXIT_FAILURE;
    }
    spectrums = (float*)malloc(envi_properties.bands * n_spectrums * sizeof(float));
    names = new string[n_spectrums];
    int spectrum_index = 0;

    if(Analyzer_tools::read_spectrums(analyzer_properties.specrums_folder_path, spectrums, names, envi_properties, &spectrum_index)) {
        cerr << "ERROR: reading spectrums. Aborting..." << endl;
        free(img);
        free(spectrums);
        delete[] names;
        return EXIT_FAILURE;
    }
    cout << "Spectrums files read with no errors." << endl;
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    ////////////////////////////////Initialize////////////////////////////////
    Analyzer_tools::Analyzer_properties analyzer_properties;
    sycl::queue device_q;
    if(initialize(analyzer_properties, device_q, argc, argv))
        return EXIT_FAILURE;

    variant<float*, sycl::buffer<float, 1>> img_d;
    variant<float*, sycl::buffer<float, 1>> spectrums_d;
    variant<float*, sycl::buffer<float, 1>> results_d;


    /////////////////////////////////Read .hdr////////////////////////////////
    ENVI_reader::ENVI_properties envi_properties;
    if(read_hdr(analyzer_properties, envi_properties))
        return EXIT_FAILURE;
    

    ///////////////////////Read hyperspectral image//////////////////////////
    float* img_h;
    if(read_hyperspectral(analyzer_properties, envi_properties, img_h))
        return EXIT_FAILURE;


    ///////////////////////Copy hyperspectral image//////////////////////////
    optional<sycl::event> opt_img_copied;
    if(Analyzer_tools::copy_to_device(analyzer_properties.USE_ACCESSORS, device_q, img_d, img_h, envi_properties.get_image_3Dsize(), &opt_img_copied))
        return EXIT_FAILURE;
    

    ////////////////////////////Read spectrums////////////////////////////////
    size_t n_spectrums;
    float* spectrums_h;
    string* names;
    if(read_spectrums(analyzer_properties, envi_properties, img_h, n_spectrums, spectrums_h, names))
        return EXIT_FAILURE;


    ////////////////////////////Copy spectrums////////////////////////////////
    optional<sycl::event> opt_spectrums_copied;
    if(Analyzer_tools::copy_to_device(analyzer_properties.USE_ACCESSORS, device_q, spectrums_d, spectrums_h, n_spectrums * envi_properties.bands, &opt_spectrums_copied))
        return EXIT_FAILURE;


    //////////////////////////////Scale img///////////////////////////////////
    sycl::event img_scaled;
    sycl::range<1> scaler_range(envi_properties.get_image_3Dsize());
    Analyzer_tools::launch_kernel_with_variants<Functors::ImgScaler, sycl::range<1>, variant<float*, sycl::buffer<float, 1>>>
                    (device_q, scaler_range, opt_spectrums_copied, img_d, envi_properties.reflectance_scale_factor);

    /////////////////////////////launch kernel////////////////////////////////
    size_t results_size = envi_properties.samples * envi_properties.lines;
    float* results_h = (float*)malloc(envi_properties.get_image_2Dsize() * sizeof(float));
    for(int i = 0; i < results_size; i++) 
        results_h[i] = 0;
    

    free_malloc_resources(opt_spectrums_copied.value(), img_h, spectrums_h, results_h);
    device_q.wait();
    if(!analyzer_properties.USE_ACCESSORS)
        free_USM_resources(device_q, img_d, spectrums_d);

    return EXIT_SUCCESS;
}