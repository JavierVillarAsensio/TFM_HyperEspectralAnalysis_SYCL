#include <config_test.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <array>

using namespace std;

#define DO_NOT_SET_VALUES false

Analyzer_tools::Analyzer_properties analyzer_properties;
float* img_h = nullptr;
float* spectrums_h = nullptr;
float* final_results_h = nullptr;
Analyzer_variant img_d;
Analyzer_variant spectrums_d;
string* names = nullptr;
sycl::queue device_q;
optional<sycl::event> copied_event;

void free_resources() {
    free(img_h);
    free(spectrums_h);
    delete[] names;
    sycl::free(get<float*>(img_d), device_q);
    sycl::free(get<float*>(spectrums_d), device_q);
}

template<typename T>
void initialize_pointer(T*& ptr, size_t ptr_size, bool set_values = true, float value = FLOAT_MAX) {
    ptr = (float*)malloc(ptr_size * sizeof(T));

    if(set_values)
        for(size_t i = 0; i < ptr_size; i++)
            ptr[i] = value;
}

exit_code check_result_img(float* ptr) {
    for(size_t i = 0; i < IMG_2D_SIZE; i++)
        if(static_cast<int>(ptr[i]) != ((i + 1)  % 2))
            return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

template<typename T>
exit_code check_scaled(T a) {
    float epsilon = 0.0001;
    size_t full_line_bands = TEST_BANDS * TEST_SAMPLES;

    for(size_t i = 0; i < TEST_IMG_SIZE; i++) {
        if(!(fabs(a[i] - TEST_SCALED_IMG[ ((i%TEST_SAMPLES) * TEST_BANDS) + ((i/TEST_SAMPLES)%TEST_BANDS) + ((i/full_line_bands) * full_line_bands) ]) < epsilon))
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

exit_code write_test_img_bil(){
    short int bil_image[TESTING_IMG_N_ELEMENTS];
    FILE *file = fopen(TEST_IMG_FILE_PATH, "wb");
    if (file == NULL){
        cout << "Error opening testing file to write. Aborting..." << endl;
        return EXIT_FAILURE;
    }

    //change to bilS
    int line_length = TEST_BANDS * TEST_SAMPLES, calculated_index;
    for(int index = 0; index < TESTING_IMG_N_ELEMENTS; index++){
        //                              column offset                               band offset                                  line offset
        calculated_index = (int)((index%TEST_SAMPLES)*TEST_BANDS)   +   (int)((index/TEST_SAMPLES)%TEST_BANDS)   +   (int)((index/line_length)*line_length);
        bil_image[index] = TESTING_IMG[calculated_index];
    }

    fwrite(bil_image, sizeof(short int), TESTING_IMG_N_ELEMENTS, file);

    fclose(file);

    return EXIT_SUCCESS;
}

template<typename Callback, typename... Args>
void test(Callback do_test, const string str, int &tests_passed, int &tests_done, Args&&... args){
    tests_done++;
    exit_code code = do_test(std::forward<Args>(args)...);
    cout << "#" << tests_done << " Testing: "  << str << " = " << TEST_RESULTS[code] << endl;;
    if (!code)
        tests_passed++;
}

/////////////////////////////ANALYZER TESTS/////////////////////////////
exit_code test_get_filename_by_extension(const char* path, const char* extension) {
    return Analyzer_tools::get_filename_by_extension(path, extension).empty() ? EXIT_FAILURE : EXIT_SUCCESS;
}

exit_code test_get_filename_by_extension_nonexistent(const char* wrong_path, const char* extension) {
    return Analyzer_tools::get_filename_by_extension(wrong_path, extension).empty() ? EXIT_SUCCESS : EXIT_FAILURE;
}

exit_code test_get_analyzer_properties(int argc, char* argv[]) {
    analyzer_properties = Analyzer_tools::initialize_analyzer(argc, argv);
    if (analyzer_properties.image_hdr_folder_path != nullptr && analyzer_properties.specrums_folder_path != nullptr)
        return EXIT_SUCCESS;
    else
        return EXIT_FAILURE;
}

exit_code test_get_wrong_analyzer_properties() {
    int fake_argc = 0;
    char** fake_argv;
    Analyzer_tools::Analyzer_properties fake_analyzer_properties = Analyzer_tools::initialize_analyzer(fake_argc, fake_argv);
    if (fake_analyzer_properties.image_hdr_folder_path != nullptr && fake_analyzer_properties.specrums_folder_path != nullptr)
        return EXIT_FAILURE;
    else
        return EXIT_SUCCESS;
}

void analyzer_tests(int& tests_done, int& tests_passed, int argc, char* argv[]) {
    test(test_get_filename_by_extension, "get filename with directory and extension", tests_passed, tests_done, TEST_HDR_IMG_DIRECTORY, HDR_FILE_EXTENSION);
    test(test_get_filename_by_extension_nonexistent, "control non-existent file in given directory and extension", tests_passed, tests_done, "a", HDR_FILE_EXTENSION);
    test(test_get_analyzer_properties, "get analyzer properties", tests_passed, tests_done, argc, argv);
    test(test_get_wrong_analyzer_properties, "bad analyzer properties are controlled", tests_passed, tests_done);
}


///////////////////////////////HDR TESTS///////////////////////////////
exit_code test_read_hdr(const char *filename) { return ENVI_reader::read_hdr(filename, analyzer_properties.envi_properties); }

exit_code test_check_wrong_hdr() {
    int temp_samples = analyzer_properties.envi_properties.samples;
    analyzer_properties.envi_properties.samples = FAILURE;
    exit_code code = ENVI_reader::check_properties(analyzer_properties.envi_properties);
    analyzer_properties.envi_properties.samples = temp_samples;
    return  code ? EXIT_SUCCESS : EXIT_FAILURE;
}

exit_code test_check_correct_values_of_hdr() {
    float epsilon = 1e-6f;
    if (analyzer_properties.envi_properties.bands != TEST_BANDS)
        return EXIT_FAILURE;
    else if (analyzer_properties.envi_properties.data_type_size != TEST_DATA_TYPE_SIZE)
        return EXIT_FAILURE;
    else if (analyzer_properties.envi_properties.header_offset != TEST_HEADER_OFFSET)
        return EXIT_FAILURE;
    else if (analyzer_properties.envi_properties.interleave != TEST_INTERLEAVE)
        return EXIT_FAILURE;
    else if (analyzer_properties.envi_properties.lines != TEST_LINES)
        return EXIT_FAILURE;
    else if (analyzer_properties.envi_properties.reflectance_scale_factor != TEST_REFLECTANCE_SCALE_FACTOR)
        return EXIT_FAILURE;
    else if (analyzer_properties.envi_properties.samples != TEST_SAMPLES)
        return EXIT_FAILURE;
    else if (analyzer_properties.envi_properties.wavelength_unit != TEST_WAVELENGTH_UNITS)
        return EXIT_FAILURE;
    
    for (size_t i = 0; i < TEST_BANDS; i++)
        if(fabs(analyzer_properties.envi_properties.wavelengths[i] - TEST_WAVELENGTHS[i]) > epsilon)
            return EXIT_FAILURE;
    
    return EXIT_SUCCESS;
}

void hdr_tests(int& tests_done, int& tests_passed){
    test(test_read_hdr, "read of .hdr ENVI header file", tests_passed, tests_done, TEST_HDR_FILE_PATH);
    test(test_check_wrong_hdr, "check wrong value from ENVI properties structure", tests_passed, tests_done);
    test(test_check_correct_values_of_hdr, "check the .hdr values are read correctly", tests_passed, tests_done);
}


///////////////////////////////IMG TESTS///////////////////////////////
exit_code test_read_img_bil() {
    img_h = (float*)malloc(analyzer_properties.envi_properties.get_image_3Dsize() * sizeof(float));
    return ENVI_reader::read_img_bil(img_h, analyzer_properties.envi_properties, TEST_IMG_FILE_PATH);
}

exit_code test_read_nonexistent_img() { return ENVI_reader::read_img_bil(nullptr, analyzer_properties.envi_properties, "a") ? EXIT_SUCCESS : EXIT_FAILURE; }

exit_code test_read_img_with_wrong_length() {
    analyzer_properties.envi_properties.bands++;
    exit_code code = ENVI_reader::read_img_bil(nullptr, analyzer_properties.envi_properties, TEST_IMG_FILE_PATH);
    analyzer_properties.envi_properties.bands--;
    return code ? EXIT_SUCCESS : EXIT_FAILURE;
}

exit_code test_img_read_correct() {
    float epsilon = 0.00001;
    size_t full_line_bands = analyzer_properties.envi_properties.bands * analyzer_properties.envi_properties.samples, samples = analyzer_properties.envi_properties.samples, bands = analyzer_properties.envi_properties.bands;
    
    for(size_t i = 0; i < analyzer_properties.envi_properties.get_image_3Dsize(); i++) 
        if (!(fabs(TESTING_IMG[ ((i%samples) * bands) + ((i/samples)%bands) + ((i/full_line_bands) * full_line_bands) ] - img_h[i]) < epsilon))
            return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

void img_tests(int& tests_done, int& tests_passed) {
    test(test_read_img_bil, "read of image with BIL interleave", tests_passed, tests_done);
    test(test_read_nonexistent_img, "control read of nonexistent img", tests_passed, tests_done);
    test(test_read_img_with_wrong_length, "read of img with length different from the .hdr file", tests_passed, tests_done);
    test(test_img_read_correct, "img read correctly", tests_passed, tests_done);
}


////////////////////////////SPECTRUMS TESTS////////////////////////////
exit_code test_count_spectrums() { 
    size_t n_spectrums = Analyzer_tools::count_spectrums(TEST_SPEC_FILE_PATH);
    analyzer_properties.n_spectrums = n_spectrums;
    return n_spectrums == N_TEST_SPECTRUM_FILES ? EXIT_SUCCESS : EXIT_FAILURE;
}

exit_code test_count_spectrums_nonexistent_path() { return Analyzer_tools::count_spectrums("a") == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }

exit_code test_read_spectrums() {
    spectrums_h = (float*)malloc(analyzer_properties.envi_properties.bands * N_TEST_SPECTRUM_FILES * sizeof(float));
    names = new string[N_TEST_SPECTRUM_FILES];
    int read_index = 0;
    return Analyzer_tools::read_spectrums(TEST_SPEC_FILE_PATH, spectrums_h, names, analyzer_properties.envi_properties, &read_index);
}

exit_code test_spectrums_read_correct() {
    float epsilon = 0.00001, diff10, diff20;
    for (size_t i = 0; i < analyzer_properties.envi_properties.bands * N_TEST_SPECTRUM_FILES; i++){
        diff10 = fabs(spectrums_h[i] - TEST_SPECTRUMS_CORRECT_REFLECTANCES[0]);
        diff20 = fabs(spectrums_h[i] - TEST_SPECTRUMS_CORRECT_REFLECTANCES[1]);
        if (!(diff10 < epsilon || diff20 < epsilon))
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;        
}

void spectrums_tests(int& tests_done, int& tests_passed) {
    test(test_count_spectrums, "count the number of spectrums", tests_passed, tests_done);
    test(test_count_spectrums_nonexistent_path, "control nonexistent path when counting spectrums", tests_passed, tests_done);
    test(test_read_spectrums, "read spectrums", tests_passed, tests_done);
    test(test_spectrums_read_correct, "read spectrums correctly", tests_passed, tests_done);
}


//////////////////////////////SYCL TESTS///////////////////////////////
exit_code test_copy_USM() { return Analyzer_tools::copy_to_device(false, device_q, img_d, img_h, analyzer_properties.envi_properties.get_image_3Dsize(), &copied_event); }

exit_code test_scale_img_USM() {
    size_t img_size = analyzer_properties.envi_properties.get_image_3Dsize();
    float* ptr_h = (float*)malloc(img_size * sizeof(float));

    Analyzer_tools::launch_kernel<Functors::ImgScaler>(device_q, copied_event, analyzer_properties, array{img_d}, analyzer_properties.envi_properties.reflectance_scale_factor).wait();

    Analyzer_tools::copy_from_device(false, device_q, ptr_h, img_d, img_size, &copied_event);
    copied_event.value().wait();

    exit_code ret = check_scaled(ptr_h);

    free(ptr_h);

    return ret;
}

exit_code test_copy_buff() {
    Analyzer_variant buff;
    return Analyzer_tools::copy_to_device(true, device_q, buff, img_h, analyzer_properties.envi_properties.get_image_3Dsize()); 
}

exit_code test_scale_img_acc() {
    size_t img_size = analyzer_properties.envi_properties.get_image_3Dsize();
    Analyzer_variant img_d_buff;
    float* ptr_h = (float*)malloc(img_size * sizeof(float));

    Analyzer_tools::copy_to_device(true, device_q, img_d_buff, img_h, img_size, &copied_event);

    Analyzer_tools::launch_kernel<Functors::ImgScaler>(device_q, copied_event, analyzer_properties, array{img_d_buff}, analyzer_properties.envi_properties.reflectance_scale_factor).wait();
    
    Analyzer_tools::copy_from_device(true, device_q, ptr_h, img_d_buff, img_size, &copied_event);
    copied_event.value().wait();

    bool ret = check_scaled(ptr_h);

    free(ptr_h);

    return ret;
}

exit_code test_initializa_SYCL_queue() { return Analyzer_tools::initialize_SYCL_queue(analyzer_properties, device_q); }

exit_code test_initialize_default_SYCL_queue() {
    analyzer_properties.device = Analyzer_tools::ACCELERATOR;
    sycl::queue test_q;
    exit_code code = Analyzer_tools::initialize_SYCL_queue(analyzer_properties, test_q);
    analyzer_properties.device = Analyzer_tools::DEFAULT;
    return code;
}

void sycl_tests(int& tests_done, int& tests_passed) {
    test(test_initializa_SYCL_queue, "initialize SYCL queue", tests_passed, tests_done);
    test(test_initialize_default_SYCL_queue, "initialize SYCL default queue when selected is not available", tests_passed, tests_done);
    test(test_copy_USM, "copy the image to device with USM", tests_passed, tests_done);
    test(test_scale_img_USM, "scale the image to normalize with USM", tests_passed, tests_done);
    test(test_copy_buff, "copy the image to device with buffers", tests_passed, tests_done);
    //test(test_scale_img_acc, "scale the image to normalize with accessors", tests_passed, tests_done);
}


/////////////////////////////KERNEL TESTS//////////////////////////////

exit_code test_basic_euclidean() {
    bool temp_ND = analyzer_properties.ND_kernel;
    analyzer_properties.ND_kernel = false;

    size_t img_2Dsize = analyzer_properties.envi_properties.get_image_2Dsize();
    size_t results_size = Functors::Euclidean<float*>::get_results_size(analyzer_properties.envi_properties.lines, 
                                                                               analyzer_properties.envi_properties.samples, 
                                                                               analyzer_properties.envi_properties.bands, 
                                                                               N_TEST_SPECTRUM_FILES, 
                                                                               analyzer_properties.ND_kernel);

    initialize_pointer(final_results_h, results_size);
    
    Analyzer_variant results_d = sycl::malloc_device<float>(results_size, device_q);

    Analyzer_tools::copy_to_device(false, device_q, spectrums_d, spectrums_h, N_TEST_SPECTRUM_FILES * analyzer_properties.envi_properties.bands, &copied_event);
    copied_event.value().wait();
    Analyzer_tools::copy_to_device(false, device_q, results_d, final_results_h, results_size, &copied_event);
    
    Analyzer_tools::launch_kernel<Functors::Euclidean>(device_q, copied_event, analyzer_properties, array{img_d, spectrums_d, results_d}, 
                                                       analyzer_properties.n_spectrums,
                                                       analyzer_properties.envi_properties.lines,
                                                       analyzer_properties.envi_properties.samples,
                                                       analyzer_properties.envi_properties.bands,
                                                       analyzer_properties.coalescent_read_size).wait();
        
    float* result_img = (float*)malloc(img_2Dsize * sizeof(float));
    Analyzer_tools::copy_from_device(false, device_q, final_results_h, results_d, img_2Dsize * 2, &copied_event);
    copied_event.value().wait();

    exit_code return_value = check_result_img(final_results_h);
    
    sycl::free(get<float*>(results_d), device_q);
    free(result_img);

    analyzer_properties.ND_kernel = temp_ND;
    return return_value;
}

exit_code test_ND_euclidean() {
    size_t temp_local_mem = analyzer_properties.device_local_memory;
    analyzer_properties.device_local_memory = 0;
    size_t results_size = Functors::Euclidean<float*>::get_results_size(analyzer_properties.envi_properties.lines, 
                                                                               analyzer_properties.envi_properties.samples, 
                                                                               analyzer_properties.envi_properties.bands, 
                                                                               N_TEST_SPECTRUM_FILES, 
                                                                               analyzer_properties.ND_kernel);

    float* results_h;
    initialize_pointer(results_h, results_size);
    
    Analyzer_variant results_d = sycl::malloc_device<float>(results_size, device_q);

    Analyzer_tools::copy_to_device(false, device_q, results_d, results_h, results_size, &copied_event);
    
    Analyzer_tools::launch_kernel<Functors::Euclidean>(device_q, copied_event, analyzer_properties, array{img_d, spectrums_d, results_d}, 
                                                       analyzer_properties.n_spectrums,
                                                       analyzer_properties.envi_properties.lines,
                                                       analyzer_properties.envi_properties.samples,
                                                       analyzer_properties.envi_properties.bands,
                                                       analyzer_properties.coalescent_read_size).wait();

    Analyzer_tools::copy_from_device(false, device_q, results_h, results_d, results_size, &copied_event);
    copied_event.value().wait();

    exit_code return_value = check_result_img(results_h);
    
    sycl::free(get<float*>(results_d), device_q);
    free(results_h);

    analyzer_properties.device_local_memory = temp_local_mem;
    return return_value;
}

exit_code test_ND_CCM() {
    size_t temp_local_mem = analyzer_properties.device_local_memory;
    analyzer_properties.device_local_memory = 0;
    size_t results_size = Functors::CCM<float*>::get_results_size(analyzer_properties.envi_properties.lines, 
                                                                               analyzer_properties.envi_properties.samples, 
                                                                               analyzer_properties.envi_properties.bands, 
                                                                               N_TEST_SPECTRUM_FILES, 
                                                                               analyzer_properties.ND_kernel);

    float* results_h;
    initialize_pointer(results_h, results_size, DO_NOT_SET_VALUES);
    
    Analyzer_variant results_d = sycl::malloc_device<float>(results_size, device_q);

    Analyzer_tools::copy_to_device(false, device_q, results_d, results_h, results_size, &copied_event);
    
    Analyzer_tools::launch_kernel<Functors::CCM>(device_q, copied_event, analyzer_properties, array{img_d, spectrums_d, results_d}, 
                                                       analyzer_properties.n_spectrums,
                                                       analyzer_properties.envi_properties.lines,
                                                       analyzer_properties.envi_properties.samples,
                                                       analyzer_properties.envi_properties.bands,
                                                       analyzer_properties.coalescent_read_size).wait();

    Analyzer_tools::copy_from_device(false, device_q, results_h, results_d, results_size, &copied_event);
    copied_event.value().wait();

    for(size_t i = 0; i < IMG_2D_SIZE; i++)
        cout << results_h[i] << " " ;
    cout << endl;

    exit_code return_value = check_result_img(results_h);
    
    sycl::free(get<float*>(results_d), device_q);
    free(results_h);

    analyzer_properties.device_local_memory = temp_local_mem;
    return return_value;
}

exit_code test_ND_localMem_euclidean() {
    size_t results_size = Functors::Euclidean<float*>::get_results_size(analyzer_properties.envi_properties.lines, 
                                                                               analyzer_properties.envi_properties.samples, 
                                                                               analyzer_properties.envi_properties.bands, 
                                                                               N_TEST_SPECTRUM_FILES, 
                                                                               analyzer_properties.ND_kernel);

    float* results_h;
    initialize_pointer(results_h, results_size);
    
    Analyzer_variant results_d = sycl::malloc_device<float>(results_size, device_q);

    Analyzer_tools::copy_to_device(false, device_q, results_d, results_h, results_size, &copied_event);
    
    Analyzer_tools::launch_kernel<Functors::Euclidean>(device_q, copied_event, analyzer_properties, array{img_d, spectrums_d, results_d}, 
                                                       analyzer_properties.n_spectrums,
                                                       analyzer_properties.envi_properties.lines,
                                                       analyzer_properties.envi_properties.samples,
                                                       analyzer_properties.envi_properties.bands,
                                                       analyzer_properties.coalescent_read_size).wait_and_throw();

    Analyzer_tools::copy_from_device(false, device_q, results_h, results_d, results_size, &copied_event);
    copied_event.value().wait();

    exit_code return_value = check_result_img(results_h);
    sycl::free(get<float*>(results_d), device_q);
    free(results_h);

    return return_value;
}

void kernel_tests(int& tests_done, int& tests_passed) {
    test(test_basic_euclidean, "basic euclidean kernel", tests_passed, tests_done);
    test(test_ND_euclidean, "ND euclidean kernel", tests_passed, tests_done);
    test(test_ND_CCM, "ND CCM kernel", tests_passed, tests_done);
    //test(test_ND_localMem_euclidean, "ND with local memory euclidean kernel", tests_passed, tests_done);
}


/////////////////////////////RESULT TESTS//////////////////////////////
exit_code test_create_results() {
    size_t img_2Dsize = analyzer_properties.envi_properties.get_image_2Dsize();

    int *nearest_materials_image = (int*)malloc(img_2Dsize * sizeof(int));
    for(size_t i = 0; i < img_2Dsize; i++)
        nearest_materials_image[i] = (int)final_results_h[i];

    exit_code code = create_results(TEST_RESULT_FILE, nearest_materials_image, analyzer_properties.envi_properties.samples, analyzer_properties.envi_properties.lines, names, analyzer_properties.n_spectrums);
    free(nearest_materials_image);
    return code;
}

void results_tests(int& tests_done, int& tests_passed) {
    test(test_create_results, "create results files", tests_passed, tests_done);
}


/////////////////////////////////MAIN//////////////////////////////////
int main(int argc, char **argv){
    int tests_done = 0, tests_passed = 0;
    if(write_test_img_bil()){
        cerr << "An unexpected error ocurred writing the test image" << endl;
        return EXIT_FAILURE;
    }

    cout << "Initiating unit tests, error messages may be shown but the test results are correct." << endl;

    analyzer_tests(tests_done, tests_passed, argc, argv);
    hdr_tests(tests_done, tests_passed);
    img_tests(tests_done, tests_passed);
    spectrums_tests(tests_done, tests_passed);
    sycl_tests(tests_done, tests_passed);
    kernel_tests(tests_done, tests_passed);
    results_tests(tests_done, tests_passed);

    if (tests_passed != tests_done)
        cout << "\033[31mThe number of tests passed is lower than the tests done: \033[0m" << "Tests passed: " << tests_passed << " < " "Tests done: " << tests_done << endl;
    else
        cout << "\033[32mThe number of tests passed matches the tests done: \033[0m" << "Tests passed: " << tests_passed << " == " "Tests done: " << tests_done << endl;

    free_resources();

    return EXIT_SUCCESS;
}