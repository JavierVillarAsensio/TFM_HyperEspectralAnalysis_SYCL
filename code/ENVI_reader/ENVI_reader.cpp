#include "ENVI_reader.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>



//constants
#define PERCENTAGE_REFACTOR 100

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
#define SPECTRUM_FIRST_VALUE_FIELD "First X Value"
#define SPECTRUM_LAST_VALUE_FIELD "Last X Value" 
#define SPEC_WAVELENGTH_UNIT_FIELD "X Units"



using namespace std;
using namespace ENVI_reader;

const unordered_map<string, int> data_type_mapper = {  //missing types not implemented
    {"1",  1},    //8-bit unsigned int
    {"2",  2},    //16-bit signed int
    {"3",  4},    //32-bit signed int
    {"4",  4},    //32-bit float
    {"5",  8},    //64-bit double
    {"12", 2},   //16-bit unsigned int
    {"13", 4},   //32-bit unsigned long
    {"14", 8},   //64-bit long signed int
    {"15", 8}    //64-bit unsigned long int
};

const unordered_map<string, int> wavelength_unit_mapper = {  //units per meter
    {"Meters",              1},
    {"m",                   1},
    {"Centimeters",       100},
    {"cm",                100},
    {"Millimeters",      1000},
    {"mm",               1000},
    {"Micrometers",   1000000},  //1e+6
    {"um",            1000000},  //1e+6
    {"Nanometers", 1000000000},  //1e+9
    {"nm",         1000000000},  //1e+9
    {"Angstroms", 10000000000}   //1e+10
};



int ENVI_map(const string str, unordered_map<string, int> mapper){
    unordered_map<string, int>::const_iterator map = mapper.find(str);

    if(map == mapper.end())
        cout << "Could not find a supported value for the property \"" << str << "\" in the .hdr ENVI header file" << endl;
    else
        return map->second;

    return -1;
}

Interleave map_interleave(const string str) {
    if (str == "bsq")
        return BSQ;
    else if(str == "bil")
        return BIL;
    else if(str == "bip")
        return BIP;
    else {
        cout << "Could not match an interleave supported type for \"" << str << "\" in the .hdr ENVI header file" << endl;
        return ERROR;
    }
}

int stoi_wrapper(const string &s){return stoi(s);}
float stof_wrapper(const string &s){return stof(s);}

template <typename Callback, typename... Args>
auto extract_value(istringstream &lineStream, Callback cb, bool mapping, Args&&... args) {
    string value;
    getline(lineStream, value, '=');

    if(mapping)
        value.erase(remove(value.begin(), value.end(), ' '), value.end());

    return cb(value, forward<Args>(args)...);
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
    exit_code check_properties(const ENVI_properties &properties){
        if (properties.wavelength_unit == FAILURE)
            return FAILURE;
        else if (properties.data_type_size == FAILURE)
            return FAILURE;
        else if (properties.interleave == ERROR)
            return FAILURE;
        else if (properties.samples == FAILURE)
            return FAILURE;
        else if (properties.lines == FAILURE)
            return FAILURE;
        else if (properties.bands == FAILURE)
            return FAILURE;
        else if (properties.header_offset == FAILURE)
            return FAILURE;
        else if (properties.reflectance_scale_factor == FAILURE)
            return FAILURE;
        else if (properties.wavelengths == nullptr)
            return FAILURE;
        else return EXIT_SUCCESS;
    }

    /**
     * @brief Returns a struct with the properties of a .hdr ENVI header file.
     * 
     * Given a file path the function returns an struct with the properties of a .hdr ENVI header file.
     * If any of the properties cannot be read it is not initialized.
     * 
     * @param filename Path to .hdr file
     * @return ENVI_properties struct or nullopt if file could not be found
     */
    optional<const ENVI_properties> read_hdr(const char* filename) {
        ifstream file(filename);

        if(!file.is_open() && !file.good()){
            cerr << "Error opening the hdr file." << endl;
            return nullopt;
        }

        struct ENVI_properties properties;

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
                    properties.wavelengths = (float*)malloc(properties.bands * sizeof(float));
                    *(properties.wavelengths) = *wavelengths;
                    free(wavelengths);
                }

                else if(key == CHANNELS_FIELD){
                    properties.bands = extract_value(lineStream, stoi_wrapper, false);                
                    wavelengths = (float*)malloc(sizeof(float) * properties.bands);
                }
                else if(key == ROWS_FIELD)
                    properties.lines = extract_value(lineStream, stoi_wrapper, false);                

                else if(key == COLS_FIELD)
                    properties.samples = extract_value(lineStream, stoi_wrapper, false);                

                else if(key == HEADER_OFFSET_FIELD)                    
                    properties.header_offset = extract_value(lineStream, stoi_wrapper, false);                

                else if(key == WAVELENGTH_UNIT_FIELD)
                    properties.wavelength_unit = extract_value(lineStream, ENVI_map, true, wavelength_unit_mapper);

                else if(key == DATA_TYPE_FIELD)                    
                    properties.data_type_size = extract_value(lineStream, ENVI_map, true, data_type_mapper);

                else if(key == REFLECTANCE_SCALE_FACTOR_FIELD)                     
                    properties.reflectance_scale_factor = extract_value(lineStream, stof_wrapper, false);                

                else if(key == INTERLEAVE_FIELD)                                         
                    properties.interleave = extract_value(lineStream, map_interleave, true);                
            }
        }

        if (check_properties(properties) == FAILURE)
            return nullopt;

        return properties;
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
    exit_code read_img_bil(float *img, const ENVI_properties properties, const char* filename) {
        int n_pixels = properties.samples * properties.lines * properties.bands;

        ifstream file(filename, ios::binary);
        if(!file.is_open()){
            cout << "Error opening the img file, it could not be opened." << endl;
            return EXIT_FAILURE;
        }

        int index = 0, data_size = properties.data_type_size;
        char buffer[data_size];
        short int value;
        float refl;

        //skip header bytes
        streampos offset = streampos(properties.header_offset);
        file.seekg(offset, ios::beg);
        while(index < n_pixels){
            file.read(reinterpret_cast<char*>(&value), data_size);

            refl = static_cast<float>(value)/PERCENTAGE_REFACTOR;
            if(refl < 0)
                refl = 0;

            img[index++] = refl;
        }

        file.close();

        if (index != n_pixels){
            cout << "Error reading img file, the number of pixels read was not the expected." << endl;
            return EXIT_FAILURE;
        }
        else
            return EXIT_SUCCESS;
    }
}
