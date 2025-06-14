#include "config_test.h"
#include <iostream>

using namespace std;

int width, height, n_channels, header_offset;
size_t file_count;
string wavelength_unit_hdr;
ENVI_reader::ENVI_properties properties;

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

exit_code test_read_hdr() {
    optional<const ENVI_reader::ENVI_properties> opt_properties = ENVI_reader::read_hdr(TEST_HDR_FILE_PATH);
    properties = opt_properties.value();
    if(!ENVI_reader::check_properties(properties)){
        if (properties.wavelength_unit != TEST_WAVELENGTH_UNITS)
            return EXIT_FAILURE;
        else if (properties.data_type_size != TEST_DATA_TYPE_SIZE)
            return EXIT_FAILURE;
        else if (properties.interleave != TEST_INTERLEAVE)
            return EXIT_FAILURE;
        else if (properties.samples != TEST_SAMPLES)
            return EXIT_FAILURE;
        else if (properties.lines != TEST_LINES)
            return EXIT_FAILURE;
        else if (properties.bands != TEST_BANDS)
            return EXIT_FAILURE;
        else if (properties.header_offset != TEST_HEADER_OFFSET)
            return EXIT_FAILURE;
        else if (properties.reflectance_scale_factor != TEST_SCALE_FACTOR)
            return EXIT_FAILURE;
        else if (*(properties.wavelengths) != *TEST_WAVELENGTHS)
            return EXIT_FAILURE;
        else return EXIT_SUCCESS;
    }
    else return EXIT_FAILURE;
}

template<typename Callback>
void test(Callback do_test, const string str, int &tests_passed, int &tests_done){
    tests_done++;
    exit_code code = do_test();
    cout << "Testing " << str << ": " << TEST_RESULTS[code] << endl;;
    if (!code)
        tests_passed++;
}

int main(){
    int tests_done = 0, tests_passed = 0;
    if(write_test_img_bil()){
        cout << "An unexpected error ocurred writing the test image" << endl;
        return EXIT_FAILURE;
    }

    test(test_read_hdr, "read of .hdr ENVI header file",tests_passed, tests_done);
}