#include <Analyzer_tools.h>
#include <iostream>
#include <Functors.h>

using namespace std;

template<typename... Ptrs>
void free_malloc_resources(sycl::event event, Ptrs... ptrs) { event.wait(); ( free(ptrs), ... ); }
template<typename... Ptrs>
void free_USM_resources(sycl::queue& q, Ptrs... ptrs) { (sycl::free(ptrs, q), ...); }

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
    img = (float*)malloc(envi_properties.get_image_size() * sizeof(float));

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

exit_code copy_hyperspectral_image(Analyzer_tools::Analyzer_properties& analyzer_properties, ENVI_reader::ENVI_properties& envi_properties, sycl::queue& device_q, 
                                   float*& img, float*& img_d, sycl::event& img_copied, unique_ptr<sycl::buffer<float, 1>>& img_buff_ptr) {
    size_t img_size = envi_properties.get_image_size();
    size_t img_size_bytes = img_size * sizeof(float);

    if(analyzer_properties.USM) {
        img_d = sycl::malloc_device<float>(img_size_bytes, device_q);
        img_buff_ptr.~unique_ptr();
        img_copied = device_q.memcpy(img_d, img, img_size_bytes);
    }
    else
        img_buff_ptr = make_unique<sycl::buffer<float, 1>>(img, sycl::range<1>(img_size));

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

exit_code copy_spectrums(Analyzer_tools::Analyzer_properties& analyzer_properties, ENVI_reader::ENVI_properties& envi_properties, sycl::queue& device_q, 
                         size_t n_spectrums, float*& spectrums, float*& spectrums_d, sycl::event& img_spectrums_copied, unique_ptr<sycl::buffer<float, 1>>& spectrums_buff_ptr) {
    size_t spectrums_size = envi_properties.bands * n_spectrums;
    size_t spectrums_size_bytes = spectrums_size * sizeof(float);

    if (analyzer_properties.USM) {
        spectrums_d = sycl::malloc_device<float>(spectrums_size_bytes, device_q);
        spectrums_buff_ptr.~unique_ptr();
    }
    else
        spectrums_buff_ptr = make_unique<sycl::buffer<float, 1>>(spectrums_size);

    img_spectrums_copied = device_q.memcpy(spectrums_d, spectrums, spectrums_size_bytes);  //the q is FIFO so the img copy goes before
    return EXIT_SUCCESS;
}

exit_code scale_img(Analyzer_tools::Analyzer_properties& analyzer_properties, ENVI_reader::ENVI_properties& envi_properties, sycl::queue& device_q, 
                    float*& img_d, sycl::event& img_copied, sycl::event& img_scaled, unique_ptr<sycl::buffer<float, 1>>& img_buff_ptr) {
    if(analyzer_properties.USM){
        struct Functors::USM::ImgScaler scaler_functor(img_d, envi_properties.reflectance_scale_factor);
        optional<sycl::event> opt_img_copied(img_copied);
        img_scaled = Analyzer_tools::launch_kernel_USM(device_q, scaler_functor, envi_properties.get_image_size(), opt_img_copied);
    }
    else {
        sycl::range<1> range(envi_properties.get_image_size());

        img_scaled = device_q.submit([&](sycl::handler& h) {
            h.depends_on(img_copied);

            sycl::accessor<float, 1, sycl::access_mode::read_write> img_acc = (*img_buff_ptr).get_access<sycl::access::mode::read_write>(h);
            struct Functors::Accessors::ImgScaler f(img_acc, envi_properties.reflectance_scale_factor);
            
            h.parallel_for(range, f);
        });
    }
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    ////////////////////////////////Initialize////////////////////////////////
    Analyzer_tools::Analyzer_properties analyzer_properties;
    sycl::queue device_q;
    if(initialize(analyzer_properties, device_q, argc, argv))
        return EXIT_FAILURE;

    float* img_d = nullptr;
    float* spectrums_d = nullptr;
    unique_ptr<sycl::buffer<float, 1>> img_buff_ptr;
    unique_ptr<sycl::buffer<float, 1>> spectrums_buff_ptr;


    /////////////////////////////////Read .hdr////////////////////////////////
    ENVI_reader::ENVI_properties envi_properties;
    if(read_hdr(analyzer_properties, envi_properties))
        return EXIT_FAILURE;
    

    ///////////////////////Read hyperspectral image//////////////////////////
    float* img;
    if(read_hyperspectral(analyzer_properties, envi_properties, img))
        return EXIT_FAILURE;


    ///////////////////////Copy hyperspectral image//////////////////////////
    sycl::event img_copied;
    if(copy_hyperspectral_image(analyzer_properties, envi_properties, device_q, img, img_d, img_copied, img_buff_ptr))
        return EXIT_FAILURE;
    

    ////////////////////////////Read spectrums////////////////////////////////
    size_t n_spectrums;
    float* spectrums;
    string* names;
    if(read_spectrums(analyzer_properties, envi_properties, img, n_spectrums, spectrums, names))
        return EXIT_FAILURE;


    ////////////////////////////Copy spectrums////////////////////////////////
    sycl::event img_spectrums_copied;
    if(copy_spectrums(analyzer_properties, envi_properties, device_q, n_spectrums, spectrums, spectrums_d, img_spectrums_copied, spectrums_buff_ptr))
        return EXIT_FAILURE;


    //////////////////////////////Scale img///////////////////////////////////
    sycl::event img_scaled;
    if(scale_img(analyzer_properties, envi_properties, device_q, img_d, img_copied, img_scaled, img_buff_ptr))
        return EXIT_FAILURE;

    img_spectrums_copied.wait();
    free_malloc_resources(img_spectrums_copied, img, spectrums);
    device_q.wait();
    free_USM_resources(device_q, img_d, spectrums_d);

    return EXIT_SUCCESS;
}