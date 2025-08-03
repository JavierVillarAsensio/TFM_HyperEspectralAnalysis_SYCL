#ifndef CONFIG_TEST_H
#define CONFIG_TEST_H

#include <Analyzer_tools.h>
#include <Functors.h>
#include <string>

////////////GENERAL DATA////////////
#define TEST_IMG_FILE_PATH "code/test/jasperRidge2_R198_test/jasperRidge2_R198.img"
#define TEST_HDR_FILE_PATH "code/test/jasperRidge2_R198_test/jasperRidge2_R198.hdr"
#define TEST_HDR_IMG_DIRECTORY "code/test/jasperRidge2_R198_test/"
#define TEST_SPEC_FILE_PATH "code/test/spectrums_test/"
#define TEST_RESULT_JPG_PATH "output/result.jpg"

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
constexpr short int TESTING_IMG[TEST_BANDS * TEST_SAMPLES * TEST_LINES] = {
    900, 1100,   1900, 2100,   800, 1200,
    1800, 2200,   1000, 1000,   2000, 2000,
    800, 1200,   1900, 2100,   900, 1100
};
const size_t TESTING_IMG_N_ELEMENTS = sizeof(TESTING_IMG) / sizeof(short int);

////////////SPECTRUMS DATA////////////
constexpr float TEST_SPECTRUMS_CORRECT_REFLECTANCES[N_TEST_SPECTRUM_FILES] = {10.0, 20.0};

#endif