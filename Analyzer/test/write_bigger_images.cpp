/**
 * Use: ./bigger_writer original/folder/path new/folder/path 4
 */

#include <ENVI_reader.hpp>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include <fstream>
#include <cmath>

#define HDR_EXTENSION ".hdr"
#define IMG_EXTENSION ".img"

using namespace std;

int argv_to_integer(const std::string& str) {
    int ret = 0;

    try {
        size_t pos;
        ret = stoi(str, &pos);
    } catch (const exception& e) {
        cerr << "The number of copies per coordinate given is not a valid integer number:" << e.what() << endl;
        return ret;
    }

    return ret;
}

void create_new_image(ENVI_reader::ENVI_properties& p, float* original, float*& bigger, int& cps_x_coordinate) {
    size_t original_img_size = p.get_image_3Dsize();
    size_t new_img_size = original_img_size * cps_x_coordinate * cps_x_coordinate * sizeof(float);
    bigger = (float*)malloc(new_img_size);
    size_t line_size = p.samples * p.bands, copy_pos_original, line_init_pos = 0, copy_pos_bigger = 0;
    
    for(size_t line = 0; line < p.lines; line++) {  //copy all the lines of the original img 
        
        for(size_t line_cp = 0; line_cp < cps_x_coordinate; line_cp++) {    //copy cps_x_coordinate times the line
            copy_pos_original = line_init_pos;

            for(size_t element = 0; element < line_size; element++){   //for every element in the line
                
                for(size_t cp = 0; cp < cps_x_coordinate; cp++) {  //copy cps_x_coordinate times the element
                    bigger[copy_pos_bigger] = original[copy_pos_original];
                    copy_pos_bigger++;
                }
                
                copy_pos_original++;
            }
        }

        line_init_pos += line_size;
    }
}

void create_new_properties(ENVI_reader::ENVI_properties& original_p, ENVI_reader::ENVI_properties& bigger_p, int& cps_x_coordinate) {
    bigger_p.samples = original_p.samples * cps_x_coordinate;
    bigger_p.lines = original_p.lines * cps_x_coordinate;
    bigger_p.bands = original_p.bands;
    bigger_p.header_offset = original_p.header_offset;
    bigger_p.data_type_size = original_p.data_type_size;
    bigger_p.interleave = original_p.interleave;
    bigger_p.wavelength_unit = original_p.wavelength_unit;
    bigger_p.reflectance_scale_factor = original_p.reflectance_scale_factor;

    bigger_p.wavelengths = (float*)malloc(bigger_p.bands * sizeof(float));
    memcpy(bigger_p.wavelengths, original_p.wavelengths, bigger_p.bands * sizeof(float));
}

int write_img(ENVI_reader::ENVI_properties& bigger_p, float* new_img, string&img_filename) {
    ofstream new_img_file(img_filename, ios::binary);
    if(!new_img_file) {
        cerr << "Error creating img file." << endl;
        return EXIT_FAILURE;
    }

    try {
        uint16_t value;
        for(size_t i = 0; i < bigger_p.lines * bigger_p.samples * bigger_p.bands; i++) {
            value = static_cast<uint16_t>(round(new_img[i]));
            new_img_file.write(reinterpret_cast<const char*>(&value), sizeof(uint16_t));
        }
    } catch (const exception& e) {
        cerr << "Error writing binary img file: " << e.what() << endl;
        new_img_file.close();
        return EXIT_FAILURE;
    }

    new_img_file.close();
    return EXIT_SUCCESS;
}

int write_hdr(ENVI_reader::ENVI_properties& bigger_p, string&hdr_filename) {
    const unordered_map<size_t, string> data_type_mapper = {  //missing types not implemented
        {1,  "1"},    //8-bit unsigned int
        {2,  "2"},    //16-bit signed int
        {4,  "3"},    //32-bit signed int
        {4,  "4"},    //32-bit float
        {8,  "5"},    //64-bit double
        {2, "12"},    //16-bit unsigned int
        {4, "13"},    //32-bit unsigned long
        {8, "14"},    //64-bit long signed int
        {8, "15"}     //64-bit unsigned long int
    };

    const unordered_map<int, string> wavelength_unit_mapper = {  //units per meter
        {1,              "meters"},
        {1,                   "m"},
        {100,       "centimeters"},
        {100,                "cm"},
        {1000,      "millimeters"},
        {1000,               "mm"},
        {1000000,   "micrometers"},  //1e+6
        {1000000,            "um"},  //1e+6
        {1000000000, "nanometers"},  //1e+9
        {1000000000,         "nm"},  //1e+9
        {10000000000, "angstroms"}   //1e+10
    };

    const unordered_map<ENVI_reader::Interleave, string> interleave_mapper = {
        {ENVI_reader::BSQ, "bsq"},
        {ENVI_reader::BIL, "bil"},
        {ENVI_reader::BIP, "bip"}
    };

    ofstream new_hdr_file(hdr_filename);
    if(!new_hdr_file) {
        cerr << "Error creating hdr file." << endl;
        return EXIT_FAILURE;
    }

    try {
        new_hdr_file << "ENVI" << endl;
        new_hdr_file << "samples = " << bigger_p.samples << endl;
        new_hdr_file << "lines = " << bigger_p.lines << endl;
        new_hdr_file << "bands = " << bigger_p.bands << endl;
        new_hdr_file << "header offset = " << bigger_p.header_offset << endl;
        new_hdr_file << "data type = " << data_type_mapper.at(bigger_p.data_type_size) << endl;
        new_hdr_file << "interleave = " << interleave_mapper.at(bigger_p.interleave) << endl;
        new_hdr_file << "byte order = " << 0 << endl;
        new_hdr_file << "wavelength units = " << wavelength_unit_mapper.at(bigger_p.wavelength_unit) << endl;
        new_hdr_file << "reflectance scale factor = " << bigger_p.reflectance_scale_factor << endl;
        new_hdr_file << "wavelength = {" << endl;
        for(size_t i = 0; i < bigger_p.bands; i++) {
            new_hdr_file << to_string(bigger_p.wavelengths[i]);
            if(!(i == (bigger_p.bands - 1)))
                new_hdr_file << ",";
        }
        new_hdr_file << "}";
    } catch (const exception& e) {
        cerr << "Error writing hdr file: " << e.what() << endl;
        new_hdr_file.close();
        return EXIT_FAILURE;
    }

    new_hdr_file.close();
    return EXIT_SUCCESS;
}

int write_files(ENVI_reader::ENVI_properties& bigger_p, float* new_img, string& original_folder_path, string& new_folder_path, int cps_x_coordinate) {
    size_t pos = original_folder_path.find_last_of("/\\");
    string folder_name = original_folder_path.substr(pos + 1) + "_x" + to_string(cps_x_coordinate);
    string new_path = new_folder_path + "/" + folder_name;
    string img_filename = new_path + "/" + folder_name + IMG_EXTENSION;
    string hdr_filename = new_path + "/" + folder_name + HDR_EXTENSION;

    filesystem::create_directory(new_path);

    cout << "Writing " << img_filename << " in " << new_path << " folder..." << endl;
    if(write_img(bigger_p, new_img, img_filename))
        return EXIT_FAILURE;

    cout << "Writing " << hdr_filename << " in " << new_path << " folder..." << endl;
    if(write_hdr(bigger_p, hdr_filename))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    if(argc < 4) {
        cerr << "Error. A path to the original file, a path to where the new folder will be written and the number of copies per coordinate are needed." << endl;
        return EXIT_FAILURE;
    }

    string original_folder_path = argv[1];
    string file, original_hdr_path, original_img_path;
    for (const auto& entry : filesystem::directory_iterator(original_folder_path)) {
        if (entry.is_regular_file()) {
            file = entry.path().filename().string();
            if (file.find(HDR_EXTENSION) != string::npos)
                original_hdr_path = original_folder_path + "/" + file;
            else if (file.find(IMG_EXTENSION) != string::npos)
                original_img_path = original_folder_path + "/" + file;
        }
    }

    string new_folder_path = argv[2];


    string cps_x_coordinate_str = argv[3];
    int cps_x_coordinate = argv_to_integer(cps_x_coordinate_str);
    if(!cps_x_coordinate)
        return EXIT_FAILURE;

    
    ENVI_reader::ENVI_properties original_p;
    ENVI_reader::read_hdr(original_hdr_path, original_p);


    float* img = (float*)malloc(original_p.get_image_3Dsize() * sizeof(float));
    ENVI_reader::read_img_bil(img, original_p, original_img_path);


    ENVI_reader::ENVI_properties bigger_p;
    cout << "Creating new properties..." << endl;
    try {
        create_new_properties(original_p, bigger_p, cps_x_coordinate);
    } catch (const exception& e) {
        cerr << "Error while creating the new properties: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    cout << "New properties created." << endl;

    float* bigger;
    cout << "Creating new image..." << endl;
    try {
        create_new_image(original_p, img, bigger, cps_x_coordinate);
    } catch (const exception& e) {
        cerr << "Error while creating the new image: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    cout << "New image created." << endl;

    size_t line_size = bigger_p.samples * bigger_p.bands;
    for(size_t i = 0; i < bigger_p.get_image_3Dsize(); i++) {
        cout << bigger[i] << " ";
        if((i % line_size) == (line_size - 1))
            cout << endl;
    }

    return write_files(bigger_p, bigger, original_folder_path, new_folder_path, cps_x_coordinate);
}