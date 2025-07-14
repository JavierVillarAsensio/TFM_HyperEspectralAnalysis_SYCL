#ifndef ANALYZER_TOOLS_H
#define ANALYZER_TOOLS_H

#include <ENVI_reader.h>
#include <string>
#include <sycl/sycl.hpp>
#include <optional>

#define HDR_FILE_EXTENSION ".hdr"
#define IMG_FILE_EXTENSION ".img" 


constexpr const char* help_message = "These are the options to execute the program:\n"
                                        "\t-h Shows this message\n"
                                        "\t-a Choose the algorithm of the analyzer, the availabes are: CCM, EUCLIDEAN (EUCLIDEAN by default)\n"
                                        "\t-s Path to the folder where the spectrums are, a folder with folders is also valid "
                                        "but the regular files contained can only be spectrums, otherwise it will cause an error\n"
                                        "\t-i Path to the folder where the image and the .hdr file are. Only one of each should be"
                                        " found in this folder\n"
                                        "\t-d Choose the device selector: CPU, GPU, ACCELERATOR or DEFAULT (DEFAULT by default)\n"
                                      "If -h option the rest will be ignored and the program will be exited, the rest of the "
                                      "options are compulsory except the ones with default values.";

namespace Analyzer_tools {

    enum Analyzer_algorithms {
        EUCLIDEAN,
        CCM
    };

    enum Analyzer_devices {
        CPU,
        GPU,
        ACCELERATOR,
        DEFAULT
    };

    struct Analyzer_properties {
        const char* specrums_folder_path = nullptr;
        const char* image_hdr_folder_path = nullptr;
        Analyzer_algorithms algorithm = EUCLIDEAN;
        Analyzer_devices device = DEFAULT;
        bool ND_kernel = false;
        int ND_max_item_work_group_size = 1;
        bool USM = false;
    };                           

    Analyzer_properties initialize_analyzer(int argc, char** argv);
    size_t count_spectrums(const char* path);
    exit_code read_spectrums(const char* path, float* spectrums, std::string* names, ENVI_reader::ENVI_properties& properties, int* spectrums_index );
    std::string get_filename_by_extension(const char* directory, const char* file_extension);
    exit_code initialize_SYCL_queue(Analyzer_tools::Analyzer_properties& properties, sycl::queue& q);

    /**
     * @brief launchs a kernel of the given functor for devices with USM
     * 
     * Checks if there is a dependency to satisfy and launches a kernel of the given functor, with the given size on the given queue.
     * 
     * @param q reference to the queue attached to the device
     * @param f functor to be executed as kernel
     * @param size size of the parallelism
     * @param opt_dependency dependency of the task optional
     */
    template<typename Functor>
    sycl::event launch_kernel_USM(sycl::queue& q, Functor f, size_t size, std::optional<sycl::event> opt_dependency) {
        sycl::event launch_event;
        sycl::range<1> range(size);

        launch_event = q.submit([&](sycl::handler& h) {
        if(opt_dependency.has_value())
            h.depends_on(opt_dependency.value());    
        
        h.parallel_for(range, f);
        });

        return launch_event;
    }

    /**
     * @brief launchs a kernel of the given functor for devices with USM
     * 
     * Checks if there is a dependency to satisfy and launches a kernel of the given functor, with the given size on the given queue.
     * This launches a kernel that should be prepared for nd_range.
     * 
     * @param q reference to the queue attached to the device
     * @param f functor to be executed as kernel
     * @param size size of the parallelism
     * @param opt_dependency dependency of the task optional
     * @param range sycl nd_range object, its N value has to be known in compilation time
     */
    template<typename Functor, int N>
    sycl::event launch_kernel_USM(sycl::queue& q, Functor f, size_t size, std::optional<sycl::event> opt_dependency, sycl::nd_range<N> range) {
        sycl::event launch_event;

        launch_event = q.submit([&](sycl::handler& h) {
        if(opt_dependency.has_value())
            h.depends_on(opt_dependency.value());    
        
        h.parallel_for(range, f);
        });

        return launch_event;
    }

    /**
     * @brief launchs a kernel of the given functor for devices without USM
     * 
     * Checks if there is a dependency to satisfy and launches a kernel with the templated functor, with the given size on the given queue.
     * 
     * @param q reference to the queue attached to the device
     * @param img_buf reference to the buffer with the image
     * @param spectrums_buf reference to the buffer with the spectrums
     * @param results_buf reference to the buffer where results will be stored
     * @param size size of the parallelism
     * @param opt_dependency dependency of the task optional
     */
    template<typename Functor>
    sycl::event launch_kernel_acc(sycl::queue& q, 
                                  sycl::buffer<float, 1>& img_buf,
                                  sycl::buffer<float, 1>& spectrums_buf,
                                  sycl::buffer<float, 1>& results_buf,
                                  size_t size, 
                                  std::optional<sycl::event> opt_dependency) {

        sycl::event launch_event;
        sycl::range<1> range(size);

        launch_event = q.submit([&](sycl::handler& h) {
            if(opt_dependency.has_value())
                h.depends_on(opt_dependency.value());

            sycl::accessor<float, 1, sycl::access_mode::read_write> img_acc = img_buf.get_access<sycl::access::mode::read_write>(h);
            sycl::accessor<float, 1, sycl::access_mode::read> spectrums_acc = spectrums_buf.get_access<sycl::access::mode::read>(h);
            sycl::accessor<float, 1, sycl::access_mode::read_write> results_acc = results_buf.get_access<sycl::access::mode::read_write>(h);

            Functor f(img_acc, spectrums_acc, results_acc);
            
            h.parallel_for(range, f);
        });

        return launch_event;
    }

}

#endif