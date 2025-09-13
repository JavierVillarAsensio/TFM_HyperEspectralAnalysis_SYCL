#include <Analyzer_tools.hpp>
#include <iostream>
#include <Functors.hpp>
#include <Results_writer.hpp>
#include <chrono>

#define OUTPUT_FOLDER "output/"

//spectrum needed fields for reading Jasper Ridge files
#define SPECTRUM_N_NEEDED_FIELDS 4
#define SPECTRUM_FIRST_VALUE_FIELD "First X Value"
#define SPECTRUM_LAST_VALUE_FIELD "Last X Value" 
#define SPECTRUM_WAVELENGTH_UNIT_FIELD "X Units"
#define SPECTRUM_NAME_FIELD "Name"

#define MICROMETERS_IN_A_METER 1000000 //for jasper ridge files

using namespace std;

using CHR_time_point = chrono::high_resolution_clock::time_point;
using Duration = chrono::duration<double, std::milli>;

template<typename... Ptrs>
void free_malloc_resources(sycl::event event, Ptrs... ptrs) { event.wait(); ( free(ptrs), ... ); }
template<typename... Ptrs>
void free_USM_resources(sycl::queue& q, Ptrs... ptrs) { (sycl::free(std::get<float*>(ptrs), q), ...); }

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

string concatenate_times(CHR_time_point init, CHR_time_point start_copy_img, CHR_time_point start_scale_img, CHR_time_point start_copy_spectrums, CHR_time_point start_kernel, CHR_time_point end) {
    Duration initialization = std::chrono::duration<double>(start_copy_img - init);
    Duration img_copied = std::chrono::duration<double>(start_scale_img - start_copy_img);
    Duration img_scaled = std::chrono::duration<double>(start_copy_spectrums - start_scale_img);
    Duration spectrums_copied = std::chrono::duration<double>(start_kernel - start_copy_spectrums);
    Duration kernel_executed = std::chrono::duration<double>(end - start_kernel);
    Duration total = std::chrono::duration<double>(end - init);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    auto format = [](double val) {
        std::ostringstream tmp;
        tmp << std::fixed << std::setprecision(6) << std::setw(18) << std::setfill(' ') << val;
        return tmp.str();
    };

    oss << format(initialization.count())     
        << format(img_copied.count())         
        << format(img_scaled.count())         
        << format(spectrums_copied.count())   
        << format(kernel_executed.count())
        << format(total.count());

    return oss.str();
}

int main(int argc, char* argv[]) {
    CHR_time_point init = chrono::high_resolution_clock::now();
    ////////////////////////////////Initialize////////////////////////////////
    Analyzer_tools::Analyzer_properties analyzer_properties;
    sycl::queue device_q;
    if(Analyzer_tools::initialize(analyzer_properties, device_q, argc, argv))
        return EXIT_FAILURE;

    Analyzer_variant img_d;
    Analyzer_variant spectrums_d;
    Analyzer_variant results_d;

    /////////////////////////////////Read .hdr////////////////////////////////
    if(Analyzer_tools::read_hdr(analyzer_properties))
        return EXIT_FAILURE;
    
    ///////////////////////Read hyperspectral image//////////////////////////
    float* img_h;
    if(Analyzer_tools::read_hyperspectral(analyzer_properties, img_h))
        return EXIT_FAILURE;

    CHR_time_point start_copy_img = chrono::high_resolution_clock::now();
    ///////////////////////Copy hyperspectral image//////////////////////////
    Event_opt opt_img_copied;
    if(Analyzer_tools::copy_to_device(analyzer_properties.USE_ACCESSORS, device_q, img_d, img_h, analyzer_properties.envi_properties.get_image_3Dsize(), &opt_img_copied))
        return EXIT_FAILURE;

    if(opt_img_copied.has_value())
        opt_img_copied.value().wait();
    CHR_time_point start_scale_img = chrono::high_resolution_clock::now();
    //////////////////////////////Scale img///////////////////////////////////
    Event_opt img_scaled;
    if(opt_img_copied.has_value())
        img_scaled = opt_img_copied.value();
    Analyzer_tools::scale_image(device_q, analyzer_properties, img_d, img_scaled, true);

    if(img_scaled.has_value())
        img_scaled.value().wait();
    CHR_time_point start_copy_spectrums = chrono::high_resolution_clock::now();

    ////////////////////////////Read spectrums////////////////////////////////
    size_t n_spectrums;
    float* spectrums_h;
    string* names;

    /*
    //ENVI files
    if(Analyzer_tools::read_spectrums(analyzer_properties, n_spectrums, spectrums_h, names))
        return EXIT_FAILURE;
    */

    
    //Jasper Ridge files
    if(read_spectrums_JasperRidge(analyzer_properties, spectrums_h, names, n_spectrums))
        return EXIT_FAILURE;
    

    free(img_h);
    analyzer_properties.n_spectrums = n_spectrums;

    ////////////////////////////Copy spectrums////////////////////////////////
    Event_opt opt_spectrums_copied;
    cout << "Copying spectrums to device..." << endl;
    if(Analyzer_tools::copy_to_device(analyzer_properties.USE_ACCESSORS, device_q, spectrums_d, spectrums_h, analyzer_properties.get_spectrums_size(), &opt_spectrums_copied)) {
        cout << "ERROR: copying spectrums to device. Aborting..." << endl;
        return EXIT_FAILURE;
    }
    cout << "Spectrums copied to device with no errors." << endl;
    
    if(opt_spectrums_copied.has_value())
        opt_spectrums_copied.value().wait();
    CHR_time_point start_kernel = chrono::high_resolution_clock::now();
    
    /////////////////////////////launch kernel////////////////////////////////
    analyzer_properties.device_local_memory = 1; // local memory causes crash, needs investigation
    Event_opt kernel_finished;
    cout << "Launching analysis kernel..." << endl;
    if(Analyzer_tools::launch_analysis(analyzer_properties, device_q, img_d, spectrums_d, results_d, kernel_finished)) {
        cout << "ERROR: launching analysis kernel. Aborting..." << endl;
        return EXIT_FAILURE;
    }
    cout << "Analysis kernel launched with no errors." << endl;

    if(kernel_finished.has_value())
        kernel_finished.value().wait();
    CHR_time_point end = chrono::high_resolution_clock::now();

    /////////////////////////////write results////////////////////////////////
    size_t img_2Dsize = analyzer_properties.envi_properties.get_image_2Dsize();
    int *nearest_materials_image = (int*)malloc(img_2Dsize * sizeof(int));
    const char* output_file_name = strrchr(analyzer_properties.image_hdr_folder_path, '/') + 1;
    string output_path = OUTPUT_FOLDER + string(output_file_name);
    output_file_name = output_path.c_str();


    cout << "Copying results from device to host..." << endl;
    Event_opt results_back;
    float* final_results_h = (float*)malloc(img_2Dsize * sizeof(float));
    if(Analyzer_tools::copy_from_device(analyzer_properties.USE_ACCESSORS, device_q, final_results_h, results_d, img_2Dsize, &results_back)) {
        cout << "ERROR: copying results from device to host. Aborting..." << endl;
        free(final_results_h);
        free(nearest_materials_image);
        return EXIT_FAILURE;
    }
    cout << "Results copied from device to host with no errors." << endl;

    if(results_back.has_value())
        results_back.value().wait(); 

    for(size_t i = 0; i < img_2Dsize; i++)
        nearest_materials_image[i] = (int)final_results_h[i];

    free(final_results_h);
    free(spectrums_h);
    
    cout << "Creating results files..." << endl;
    if(create_results(output_file_name, 
                      nearest_materials_image, 
                      analyzer_properties.envi_properties.samples, 
                      analyzer_properties.envi_properties.lines, 
                      names, 
                      analyzer_properties.n_spectrums, 
                      concatenate_times(init,  start_copy_img, start_scale_img, start_copy_spectrums, start_kernel, end))) {
        cout << "ERROR: creating results files. Aborting..." << endl;
        free(nearest_materials_image);
        return EXIT_FAILURE;
    }
    cout << "Results files created with no errors." << endl;
    free(nearest_materials_image);
    delete[] names;
    
    return EXIT_SUCCESS;
}