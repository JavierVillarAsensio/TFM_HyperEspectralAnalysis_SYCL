#include <config_test.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

ENVI_reader::ENVI_properties envi_properties;
Analyzer_tools::Analyzer_properties analyzer_properties;
float* img = nullptr;
float* spectrums = nullptr;
string* names = nullptr;

void free_resources() {
    free(img);
    free(spectrums);
    delete[] names;
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
    exit_code code = do_test(forward<Args>(args)...);
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
exit_code test_read_hdr(const char *filename) { return ENVI_reader::read_hdr(filename, &envi_properties); }

exit_code test_read_fake_hdr(const char *filename) {
    ENVI_reader::ENVI_properties bad_properties;
    return ENVI_reader::read_hdr(filename, &bad_properties) ? EXIT_SUCCESS : EXIT_FAILURE;
}

exit_code test_check_wrong_hdr() {
    int temp_samples = envi_properties.samples;
    envi_properties.samples = FAILURE;
    exit_code code = ENVI_reader::check_properties(&envi_properties);
    envi_properties.samples = temp_samples;
    return  code ? EXIT_SUCCESS : EXIT_FAILURE;
}

exit_code test_check_correct_values_of_hdr() {
    float epsilon = 1e-6f;
    if (envi_properties.bands != TEST_BANDS)
        return EXIT_FAILURE;
    else if (envi_properties.data_type_size != TEST_DATA_TYPE_SIZE)
        return EXIT_FAILURE;
    else if (envi_properties.header_offset != TEST_HEADER_OFFSET)
        return EXIT_FAILURE;
    else if (envi_properties.interleave != TEST_INTERLEAVE)
        return EXIT_FAILURE;
    else if (envi_properties.lines != TEST_LINES)
        return EXIT_FAILURE;
    else if (envi_properties.reflectance_scale_factor != TEST_REFLECTANCE_SCALE_FACTOR)
        return EXIT_FAILURE;
    else if (envi_properties.samples != TEST_SAMPLES)
        return EXIT_FAILURE;
    else if (envi_properties.wavelength_unit != TEST_WAVELENGTH_UNITS)
        return EXIT_FAILURE;
    
    for (size_t i = 0; i < TEST_BANDS; i++)
        if(fabs(envi_properties.wavelengths[i] - TEST_WAVELENGTHS[i]) > epsilon)
            return EXIT_FAILURE;
    
    return EXIT_SUCCESS;
}

void hdr_tests(int& tests_done, int& tests_passed){
    test(test_read_hdr, "read of .hdr ENVI header file", tests_passed, tests_done, TEST_HDR_FILE_PATH);
    test(test_read_fake_hdr, "read of non existent .hdr ENVI header file shows error", tests_passed, tests_done, "a");
    test(test_check_wrong_hdr, "check wrong value from ENVI properties structure", tests_passed, tests_done);
    test(test_check_correct_values_of_hdr, "check the .hdr values are read correctly", tests_passed, tests_done);
}

///////////////////////////////IMG TESTS///////////////////////////////
exit_code test_read_img_bil() {
    img = (float*)malloc(envi_properties.get_image_size() * sizeof(float));
    return ENVI_reader::read_img_bil(img, &envi_properties, TEST_IMG_FILE_PATH);
}

exit_code test_read_nonexistent_img() { return ENVI_reader::read_img_bil(nullptr, &envi_properties, "a") ? EXIT_SUCCESS : EXIT_FAILURE; }

exit_code test_read_img_with_wrong_length() {
    envi_properties.bands++;
    exit_code code = ENVI_reader::read_img_bil(nullptr, &envi_properties, TEST_IMG_FILE_PATH);
    envi_properties.bands--;
    return code ? EXIT_SUCCESS : EXIT_FAILURE;
}

exit_code test_img_read_correct() {
    float epsilon = 0.00001;
    size_t full_line_bands = envi_properties.bands * envi_properties.samples, samples = envi_properties.samples, bands = envi_properties.bands;
    
    for(size_t i = 0; i < envi_properties.get_image_size(); i++) 
        if (!(fabs(TESTING_IMG[ ((i%samples) * bands) + ((i/samples)%bands) + ((i/full_line_bands) * full_line_bands) ] - img[i]) < epsilon))
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
exit_code test_count_spectrums() { return Analyzer_tools::count_spectrums(TEST_SPEC_FILE_PATH) == N_TEST_SPECTRUM_FILES ? EXIT_SUCCESS : EXIT_FAILURE; }

exit_code test_count_spectrums_nonexistent_path() { return Analyzer_tools::count_spectrums("a") == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }

exit_code test_read_spectrums() {
    spectrums = (float*)malloc(envi_properties.bands * N_TEST_SPECTRUM_FILES * sizeof(float));
    names = new string[N_TEST_SPECTRUM_FILES];
    int read_index = 0;
    return Analyzer_tools::read_spectrums(TEST_SPEC_FILE_PATH, spectrums, names, envi_properties, &read_index);
}

exit_code test_spectrums_read_correct() {
    float epsilon = 0.00001, diff10, diff20;
    for (size_t i = 0; i < envi_properties.bands * N_TEST_SPECTRUM_FILES; i++){
        diff10 = fabs(spectrums[i] - TEST_SPECTRUMS_CORRECT_REFLECTANCES[0]);
        diff20 = fabs(spectrums[i] - TEST_SPECTRUMS_CORRECT_REFLECTANCES[1]);
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


/////////////////////////////////MAIN//////////////////////////////////
int main(int argc, char **argv){
    int tests_done = 0, tests_passed = 0;
    if(write_test_img_bil()){
        cerr << "An unexpected error ocurred writing the test image" << endl;
        return EXIT_FAILURE;
    }

    analyzer_tests(tests_done, tests_passed, argc, argv);
    hdr_tests(tests_done, tests_passed);
    img_tests(tests_done, tests_passed);
    spectrums_tests(tests_done, tests_passed);

    if (tests_passed != tests_done)
        cout << "\033[31mThe number of tests passed is lower than the tests done: \033[0m" << "Tests passed: " << tests_passed << " < " "Tests done: " << tests_done << endl;
    else
        cout << "\033[32mThe number of tests passed matches the tests done: \033[0m" << "Tests passed: " << tests_passed << " == " "Tests done: " << tests_done << endl;

    free_resources();

    return EXIT_SUCCESS;
}