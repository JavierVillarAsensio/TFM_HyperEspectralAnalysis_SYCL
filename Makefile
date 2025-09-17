binary = Analyzer.exe
test_binary = Test

include_folder = ./Analyzer/include
cpp_folder = ./Analyzer/code
analyzer_JR_folder = ./Analyzer/jasperRidge2_R198
object_files_folder = ./Analyzer/object_files

JR_folder = /jasperRidge2_R198

test_folder_extension = _test
test_folder = ./Analyzer/test
test_spectrums = /spectrums_test
include_test = -I./Analyzer/test

output_folder = ./output

given_spectrums_folder = ./spectrums/given_spectrums
test_spectrums_folder = ./spectrums/test_materials

cxx = icpx -std=c++20 -g -O2 -fsycl -fno-inline -fsycl-device-code-split=per_kernel -I$(include_folder)
precompiled = $(object_files_folder)/ENVI_reader.o $(object_files_folder)/Analyzer_tools.o $(object_files_folder)/Results_writer.o
analyzer_precompiled = $(precompiled) $(object_files_folder)/Analyzer.o
test_precompiled = $(precompiled) $(object_files_folder)/tests.o

image_writer_binary = bigger_images_writer

# --------------------- precompilations --------------------- #

$(object_files_folder)/ENVI_reader.o: $(cpp_folder)/ENVI_reader.cpp $(include_folder)/ENVI_reader.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/Results_writer.o: $(cpp_folder)/Results_writer.cpp $(include_folder)/Results_writer.hpp $(include_folder)/Analyzer_tools.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/Analyzer_tools.o: $(cpp_folder)/Analyzer_tools.cpp $(include_folder)/Analyzer_tools.hpp $(include_folder)/ENVI_reader.hpp $(include_folder)/Functors.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/tests.o: $(test_folder)/tests.cpp $(include_folder)/Analyzer_tools.hpp $(include_folder)/ENVI_reader.hpp $(include_folder)/Functors.hpp $(include_folder)/Results_writer.hpp | $(object_files_folder)
	$(cxx) $(include_test) -c $< -o $@

$(object_files_folder)/Analyzer.o: $(cpp_folder)/Analyzer.cpp $(include_folder)/Analyzer_tools.hpp $(include_folder)/ENVI_reader.hpp $(include_folder)/Functors.hpp $(include_folder)/Results_writer.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/write_bigger_images.o: $(test_folder)/write_bigger_images.cpp $(include_folder)/ENVI_reader.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

# ----------------------- compilations ---------------------- #

$(test_binary): $(test_precompiled) 
	$(MAKE) clean
	$(cxx) $^ -lmatio -o $@

$(binary): $(analyzer_precompiled) 
	$(MAKE) clean
	$(cxx) $^ -lmatio -o $@

$(image_writer_binary): $(object_files_folder)/ENVI_reader.o $(object_files_folder)/write_bigger_images.o | $(object_files_folder)
	$(cxx) $^ -o $@

# --------------------------- run --------------------------- #

run_test: $(test_binary) clean_output
	./$(test_binary) -s $(test_folder)$(test_spectrums) -i $(test_folder)$(JR_folder)$(test_folder_extension) -d GPU

run_Analyzer: $(binary) clean_output
	./$(binary) -s $(test_spectrums_folder) -i $(analyzer_JR_folder)

run_Analyzer_JR: $(binary) clean_output
	./$(binary) -s $(given_spectrums_folder) -i $(analyzer_JR_folder) -a CCM -d GPU

run_image_writer: $(image_writer_binary) clean_output
	./$(image_writer_binary) $(analyzer_JR_folder) $(test_folder) 2


# -------------------------- utils -------------------------- #

$(object_files_folder):
	mkdir -p $(object_files_folder)

clean:
	rm -f $(binary)
	rm -f $(test_binary)
	rm -f *.txt
	rm -f $(output_folder)/*

clean_all: clean
	rm -rf $(object_files_folder)
	rm -rf $(output_folder)

clean_output:
	rm -f $(output_folder)/*