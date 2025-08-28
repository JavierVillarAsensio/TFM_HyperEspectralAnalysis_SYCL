#ifndef CONFIG_TEST_H
#define CONFIG_TEST_H

#include <Analyzer_tools.hpp>
#include <string>

////////////GENERAL DATA////////////
#define TEST_IMG_FILE_PATH "Analyzer/test/jasperRidge2_R198_test/jasperRidge2_R198.img"
#define TEST_HDR_FILE_PATH "Analyzer/test/jasperRidge2_R198_test/jasperRidge2_R198.hdr"
#define TEST_HDR_IMG_DIRECTORY "Analyzer/test/jasperRidge2_R198_test/"
#define TEST_SPEC_FILE_PATH "Analyzer/test/spectrums_test/"
#define TEST_RESULT_FILE "output/result"

extern const std::string ERROR = "\033[31mERROR\033[0m";   // Red
extern const std::string PASS  = "\033[32mPASS\033[0m";    // Green
extern const std::string TEST_RESULTS[2] = {PASS, ERROR};

constexpr size_t N_TEST_SPECTRUM_FILES  = 2;

typedef int exit_code;

////////////HDR DATA////////////
constexpr int TEST_SAMPLES = 3;
constexpr int TEST_LINES   = 3;
constexpr int TEST_BANDS   = 2;
constexpr int TEST_HEADER_OFFSET = 0;
constexpr int TEST_DATA_TYPE_SIZE = 2;
constexpr int TEST_WAVELENGTH_UNITS = 1000000000;
constexpr int TEST_REFLECTANCE_SCALE_FACTOR = 10000.000000;
const ENVI_reader::Interleave TEST_INTERLEAVE = ENVI_reader::BIL;
constexpr float TEST_WAVELENGTHS[TEST_BANDS] = {2000.0, 4000.0};

////////////IMAGE DATA////////////
//Euclidean test image data
constexpr size_t TEST_IMG_SIZE = TEST_BANDS * TEST_SAMPLES * TEST_LINES;
constexpr short int TESTING_IMG[TEST_IMG_SIZE] = {
    415, 786,   2648, 3138,   417, 769,
    2651, 2542,  379, 883,   2547, 2892,
    697, 413,   2664, 2779,   656, 532
};
constexpr float TEST_SCALED_IMG[TEST_IMG_SIZE] = {
    4.15, 7.86,   26.48, 31.38,   4.17, 7.69,
    26.51, 25.42,  3.79, 8.83,   25.47, 28.92,
    6.97, 4.13,   26.64, 27.79,   6.56, 5.32
};
const size_t TESTING_IMG_N_ELEMENTS = sizeof(TESTING_IMG) / sizeof(short int);
const size_t IMG_2D_SIZE = TEST_SAMPLES * TEST_LINES;

constexpr float TEST_SPECTRUMS_CORRECT_REFLECTANCES[N_TEST_SPECTRUM_FILES * TEST_BANDS] = {8.49, 13.26, 17.64, 24.91};

//CCM test image data
constexpr size_t CCM_EXAMPLE_SIZE = 4;
constexpr size_t N_TEST_CCM_SPECTRUMS = 2;
constexpr float SPECTRUMS_CCM[CCM_EXAMPLE_SIZE] = {2.0f, 3.0f, 10.0f, 2.0f};
constexpr float IMG_CCM[CCM_EXAMPLE_SIZE] = {1.0f, 10.0f, 2.0f, 1.0f};

#define CCM_CORRECT(ptr) (ptr[0] == 0 && ptr[1] == 1 ? EXIT_SUCCESS : EXIT_FAILURE)

#endif