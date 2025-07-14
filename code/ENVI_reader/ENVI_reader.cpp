#include <ENVI_reader.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <algorithm>
#include <cctype>



//hdr needed fields
#define WAVELENGTH_FIELD "wavelength"
#define ROWS_FIELD "lines"
#define COLS_FIELD "samples"
#define CHANNELS_FIELD "bands"
#define HEADER_OFFSET_FIELD "header offset"
#define WAVELENGTH_UNIT_FIELD "wavelength units"
#define DATA_TYPE_FIELD "data type"
#define REFLECTANCE_SCALE_FACTOR_FIELD "reflectance scale factor"
#define INTERLEAVE_FIELD "interleave"
#define END_FIELD "}"

//spectrum needed fields
#define SPECTRUM_N_NEEDED_FIELDS 4
#define SPECTRUM_FIRST_VALUE_FIELD "First X Value"
#define SPECTRUM_LAST_VALUE_FIELD "Last X Value" 
#define SPECTRUM_WAVELENGTH_UNIT_FIELD "X Units"
#define SPECTRUM_NAME_FIELD "Name"



using namespace std;
using namespace ENVI_reader;



//constant variables
const unordered_map<string, int> data_type_mapper = {  //missing types not implemented
    {"1",  1},    //8-bit unsigned int
    {"2",  2},    //16-bit signed int
    {"3",  4},    //32-bit signed int
    {"4",  4},    //32-bit float
    {"5",  8},    //64-bit double
    {"12", 2},    //16-bit unsigned int
    {"13", 4},    //32-bit unsigned long
    {"14", 8},    //64-bit long signed int
    {"15", 8}     //64-bit unsigned long int
};

const unordered_map<string, int> wavelength_unit_mapper = {  //units per meter
    {"meters",              1},
    {"m",                   1},
    {"centimeters",       100},
    {"cm",                100},
    {"millimeters",      1000},
    {"mm",               1000},
    {"micrometers",   1000000},  //1e+6
    {"um",            1000000},  //1e+6
    {"nanometers", 1000000000},  //1e+9
    {"nm",         1000000000},  //1e+9
    {"angstroms", 10000000000}   //1e+10
};

const unordered_map<string, Interleave> interleave_mapper = {
    {"bsq", BSQ},
    {"bil", BIL},
    {"bip", BIP}
};



//struct functions
size_t ENVI_properties::get_image_size() const noexcept { return samples * lines * bands; }
ENVI_properties::~ENVI_properties() noexcept { free(wavelengths); }



//non public functions
template <typename value>
value map(string str, unordered_map<string, value> mapper, value fallback){
    transform(str.begin(), str.end(), str.begin(), [](unsigned char c){return tolower(c);});
    typename unordered_map<string, value>::const_iterator mapped = mapper.find(str);
    return (mapped == mapper.end()) ? fallback : mapped->second;
}

int stoi_wrapper(const string &s){return stoi(s);}
float stof_wrapper(const string &s){return stof(s);}

template <typename Callback, typename... Args>
auto extract_value(istringstream &lineStream, Callback cb, bool mapping, Args&&... args) {
    string value;
    getline(lineStream, value, '=');

    if(mapping)
        value.erase(remove(value.begin(), value.end(), ' '), value.end());

    return cb(value, std::forward<Args>(args)...);
}

void save_reflectances(ifstream& file, float* reflectances, ENVI_properties& properties, bool asc_order, int wavelengths_scale_factor){
    string line;
    int wavelengths_index, iterate, reflectances_index = 0;
    float wavelength, reflectance, previous_reflectance, diff, previous_diff = 10000000;
    
    if (asc_order){
        wavelengths_index = 0;
        iterate = 1;
    }
    else {
        wavelengths_index = properties.bands -1;
        iterate = -1;
    }

    while (getline(file, line)){ 
        istringstream line_stream(line);

        line_stream >> wavelength >> reflectance;
        wavelength *= wavelengths_scale_factor;

        diff = abs(properties.wavelengths[wavelengths_index] - wavelength);
        if(previous_diff < diff){
            reflectances[reflectances_index++] = previous_reflectance;
            wavelengths_index += iterate;
        }
        
        previous_diff = diff;
        previous_reflectance = reflectance;
    }
}



namespace ENVI_reader {
    /**
     * @brief Check every property read of the struct if it is correctly initialized
     * 
     * This function only checks if the variables are initialized, not if their values are correct
     * 
     * @param properties The struct to be checked
     * @return exit_code; -1 if FAILURE, 0 if EXIT_SUCCESS
     */
    exit_code check_properties(const ENVI_properties* properties){
        if (properties->wavelength_unit == FAILURE)
            return EXIT_FAILURE;
        else if (properties->data_type_size == FAILURE)
            return EXIT_FAILURE;
        else if (properties->interleave == FAILURE)
            return EXIT_FAILURE;
        else if (properties->samples == FAILURE)
            return EXIT_FAILURE;
        else if (properties->lines == FAILURE)
            return EXIT_FAILURE;
        else if (properties->bands == FAILURE)
            return EXIT_FAILURE;
        else if (properties->header_offset == FAILURE)
            return EXIT_FAILURE;
        else if (properties->reflectance_scale_factor == FAILURE)
            return EXIT_FAILURE;
        else if (properties->wavelengths == nullptr)
            return EXIT_FAILURE;
        else return EXIT_SUCCESS;
    }

    /**
     * @brief Returns a struct with the properties of a .hdr ENVI header file.
     * 
     * Given a file path the function returns an struct with the properties of a .hdr ENVI header file.
     * If any of the properties cannot be read it is not initialized.
     * 
     * @param filename Path to .hdr file
     * @param ENVI_properties pointer to the struct where the properties will be written
     * @return exit_code type indicating EXIT_SUCCESS or EXIT_FAILURE
     */
    exit_code read_hdr(const string filename, ENVI_reader::ENVI_properties* properties) {
        ifstream file(filename);

        if(!file.is_open() && !file.good()){
            cerr << "Error opening the hdr file." << endl;
            return EXIT_FAILURE;
        }

        string line, key, value;
        bool waves_read = false;
        float* wavelengths;
        float number;
        int index = 0;

        while (getline(file, line)){
            istringstream lineStream(line);
            if (getline(lineStream, key, '=')) {
                key.erase(key.find_last_not_of(" \n\r\t")+1); //erase spaces or any dirty char

                if(key == WAVELENGTH_FIELD){
                    while (!waves_read) {
                        getline(file, line, ',');
                        stringstream numbers(line);
                        while(getline(numbers, value) && !waves_read){
                            if(value.size() > 0){
                                number = stof(value);
                                wavelengths[index] = number;
                                index++;
                                
                            }
                            if(value.find(END_FIELD) != string::npos)
                                waves_read = true;
                        }
                    }
                    properties->wavelengths = (float*)malloc(properties->bands * sizeof(float));
                    memcpy(properties->wavelengths, wavelengths, properties->bands * sizeof(float));
                    free(wavelengths);
                }

                else if(key == CHANNELS_FIELD){
                    properties->bands = extract_value(lineStream, stoi_wrapper, false);                
                    wavelengths = (float*)malloc(sizeof(float) * properties->bands);
                }
                else if(key == ROWS_FIELD)
                    properties->lines = extract_value(lineStream, stoi_wrapper, false);                

                else if(key == COLS_FIELD)
                    properties->samples = extract_value(lineStream, stoi_wrapper, false);                

                else if(key == HEADER_OFFSET_FIELD)                    
                    properties->header_offset = extract_value(lineStream, stoi_wrapper, false);

                else if(key == REFLECTANCE_SCALE_FACTOR_FIELD)
                    properties->reflectance_scale_factor = extract_value(lineStream, stoi_wrapper, false);            

                else if(key == WAVELENGTH_UNIT_FIELD)
                    properties->wavelength_unit = extract_value(lineStream, map<decltype(properties->wavelength_unit)>, true, wavelength_unit_mapper, FAILURE);

                else if(key == DATA_TYPE_FIELD)                    
                    properties->data_type_size = extract_value(lineStream, map<decltype(properties->data_type_size)>, true, data_type_mapper, FAILURE);

                else if(key == INTERLEAVE_FIELD)                                         
                    properties->interleave = extract_value(lineStream, map<decltype(properties->interleave)>, true, interleave_mapper, Interleave::FAILURE);                
            }
        }

        if (check_properties(properties)){
            cerr << "Error checking the properties struct, every variable has to have a valid value" << endl;
            return EXIT_FAILURE;
        }
            
        return EXIT_SUCCESS;
    }

    /**
     * @brief Reads image with "bil" interleave
     * 
     * Reads the image according to the properties struct and writes it in the given pointer.
     * 
     * @param img Pointer where the image will be written
     * @param properties ENVI_properties struct used to read the image properly
     * @param filename Path to the hiperespectral image
     * @return EXIT_SUCCESS or EXIT_FAILURE if any error is detected
     */
    exit_code read_img_bil(float *img, const ENVI_properties* properties, const string filename) {
        ifstream file(filename, ios::binary | ios::ate);
        if(!file.is_open()){
            cerr << "Error opening the img file, it could not be opened." << endl;
            return EXIT_FAILURE;
        }

        size_t image_size_3D = properties->get_image_size(), data_size = properties->data_type_size, index = 0;

        streamsize file_size = file.tellg();
        file.seekg(0, ios::beg);
        if (file_size != ((image_size_3D * data_size) + properties->header_offset)){
            cerr << "Error, the number of bytes of the image is not the expected" << endl;
            file.close();
            return EXIT_FAILURE;
        }

        short int data;
        float refl;
        streampos offset = streampos(properties->header_offset);
        file.seekg(offset, ios::beg);
        while(index < image_size_3D) {
            file.read(reinterpret_cast<char*>(&data), data_size);
            refl = static_cast<float>(data);
            
            if(refl < 0)
                refl = 0;

            img[index++] = refl;
        }

        file.close();

        return EXIT_SUCCESS;
    }



    /**
     * @brief Reads reflectances of a spectrum file
     * 
     * Given a spectrum file it stores its reflectances, also returns the
     * name of the spectrum and may change the properties wavelenghts to
     * match the wavelength unit of the spectrums
     * 
     * @param filename Path to the file of the spectrum
     * @param reflectances pointer to store the reflectances
     * @param name reference to the direction where the spectrum name will be stored
     * @param properties struct with the properties of the .hdr file
     */
    exit_code read_spectrum(const string filename, float* reflectances, string &name, ENVI_properties& properties){
        ifstream file(filename);
        if(!file.is_open()){
            cout << "Error opening the spectrum file, it could not be opened. Aborting." << endl;
            return(EXIT_FAILURE);
        }

        string line, segment, wavelength_unit_spec;
        int needed_fields_found = 0;
        float first_value, last_value;

        while (needed_fields_found < SPECTRUM_N_NEEDED_FIELDS) {
            getline(file, line);
            istringstream line_stream(line);
            getline(line_stream, segment, ':');

            if (segment == SPECTRUM_WAVELENGTH_UNIT_FIELD){
                getline(line_stream, segment, ':');
                size_t open = segment.find('('), close = segment.find(')');
                if (open != string::npos && close != string::npos && close > open)
                    wavelength_unit_spec = segment.substr(open + 1, close - open - 1);
                needed_fields_found++;
            }

            else if (segment == SPECTRUM_FIRST_VALUE_FIELD){
                getline(line_stream, segment, ':');
                first_value = stof(segment);
                needed_fields_found++;
            }

            else if (segment == SPECTRUM_LAST_VALUE_FIELD){
                getline(line_stream, segment, ':');
                last_value = stof(segment);
                needed_fields_found++;
            }

            else if (segment == SPECTRUM_NAME_FIELD) {
                getline(line_stream, segment, ':');
                segment.erase(segment.find_last_not_of(" \n\r\t")+1); //erase spaces or any dirty char
                name = segment;
                needed_fields_found++;
            }
        }
        
        int wavelength_scale_factor = properties.wavelength_unit / map(wavelength_unit_spec, wavelength_unit_mapper, -1);
        save_reflectances(file, reflectances, properties, first_value < last_value ? true : false, wavelength_scale_factor);

        file.close();
        return EXIT_SUCCESS;
    }
}
