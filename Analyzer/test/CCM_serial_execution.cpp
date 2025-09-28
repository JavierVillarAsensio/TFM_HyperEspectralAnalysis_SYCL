#include <ENVI_reader.hpp>
#include <Analyzer_tools.hpp>
#include <Results_writer.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>

using namespace std;
using CHR_time_point = chrono::high_resolution_clock::time_point;

const int argc = 9;
const char* argv[] = {"program", "-i", "./Analyzer/test/jasperRidge2_R198_x10", "-s", "./spectrums/given_spectrums", "-a", "CCM", "-d", "CPU"};

exit_code read_spectrums_JasperRidge(Analyzer_tools::Analyzer_properties& p, float*& reflectances, string*& names, size_t& n_spectrums){
    int spectrum_index = 0;
    names = new string[4];
    reflectances = (float*)malloc(4 * p.envi_properties.bands * sizeof(float));
    for (const auto& entry : filesystem::directory_iterator(p.specrums_folder_path)) {
        if(entry.is_regular_file()) {
            ifstream file(entry.path());
            if(!file.is_open()){
                cerr << "Error opening the spectrum file: " << entry.path() << " it could not be opened. Aborting." << endl;
                return(EXIT_FAILURE);
            }

            string line, segment;

            getline(file, line);
            istringstream line_stream(line);
            getline(line_stream, segment, ':');
            getline(line_stream, segment, ':');
            names[spectrum_index] = segment.substr(segment.find_first_not_of(" "));

            streampos pos;
            float wavelength, reflectance;

            while(true) {   //move to the reflectances section
                pos = file.tellg();

                getline(file, line);
                istringstream line_stream(line);
                if((line_stream >> wavelength >> reflectance)) {
                    file.seekg(pos);
                    break;
                }
            }

            size_t reflectances_init_pos = spectrum_index * p.envi_properties.bands;
            for(size_t i = 0; i < p.envi_properties.bands; i++) { //read reflectances
                getline(file, line);
                istringstream line_stream(line);
                line_stream >> wavelength >> reflectance;
                reflectances[i + reflectances_init_pos] = reflectance*100;
            }
        }
        spectrum_index++;
    }
    n_spectrums = spectrum_index; 

    return EXIT_SUCCESS;
}

int main() {
    float* img;
    float* spectrums;
    string* names;
    size_t n_spectrums;
    Analyzer_tools::Analyzer_properties analyzer_properties;

    analyzer_properties = Analyzer_tools::initialize_analyzer(argc, (char**)argv);
    Analyzer_tools::read_hdr(analyzer_properties);
    Analyzer_tools::read_hyperspectral(analyzer_properties, img);
    read_spectrums_JasperRidge(analyzer_properties, spectrums, names, n_spectrums);

    float* results = (float*)malloc(analyzer_properties.envi_properties.get_image_2Dsize() * sizeof(float));
    size_t n_bands = analyzer_properties.envi_properties.bands;
    size_t n_cols = analyzer_properties.envi_properties.samples;
    size_t n_rows = analyzer_properties.envi_properties.lines;
    size_t img_offset;
    float highest_correlation;
    size_t highest_correlation_index;

    CHR_time_point start = chrono::high_resolution_clock::now();
    for(size_t pixel_idx = 0; pixel_idx < analyzer_properties.envi_properties.get_image_2Dsize(); pixel_idx++) {
        img_offset = (pixel_idx / n_cols * n_bands) + (pixel_idx % n_cols);

        highest_correlation = -2; //correlation goes from -1 to 1
        highest_correlation_index = -1;

        for(size_t spectrum_idx = 0; spectrum_idx < n_spectrums; spectrum_idx++) {
            float sum_xy = 0;
            float sum_x = 0;
            float sum_y = 0;
            float sum_x2 = 0;
            float sum_y2 = 0;

            size_t spectrum_offset = spectrum_idx * n_bands;

            for(size_t band_idx = 0; band_idx < n_bands; band_idx++) {
                float x = img[img_offset + band_idx * n_cols];
                float y = spectrums[spectrum_offset + band_idx];

                sum_xy += x * y;
                sum_x += x;
                sum_y += y;
                sum_x2 += x * x;
                sum_y2 += y * y;
            }

            float numerator = n_bands * sum_xy - sum_x * sum_y;
            float denominator = sqrt((n_bands * sum_x2 - sum_x * sum_x) * (n_bands * sum_y2 - sum_y * sum_y));

            float correlation = (denominator == 0) ? 0 : numerator / denominator;

            if(correlation > highest_correlation) {
                highest_correlation = correlation;
                highest_correlation_index = spectrum_idx;
            }
        }
        results[pixel_idx] = highest_correlation_index;
    }
    CHR_time_point end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end - start;

    float a = results[0];
    float b = results[analyzer_properties.envi_properties.get_image_2Dsize()-1];
    cout << "a y b: " << a << ", " << b << endl;

    cout << "Time for CCM serial execution (ms): " << duration.count() << endl;

    free(img);
    free(spectrums);
    free(results);
    delete[] names;

    return EXIT_SUCCESS;
}