binary = Analyzer.exe
test_binary = Test

include_folder = ./Analyzer/include
cpp_folder = ./Analyzer/code
object_files_folder = ./Analyzer/object_files

JR_folder = /jasperRidge2_R198

test_folder_extension = _test
test_folder = ./Analyzer/test
test_spectrums = /spectrums_test
include_test = -I./Analyzer/test

given_spectrums_folder = ./spectrums/given_spectrums

cxx = icpx -std=c++20 -fsycl -I$(include_folder)
precompiled = $(object_files_folder)/ENVI_reader.o $(object_files_folder)/Analyzer_tools.o $(object_files_folder)/Results_writer.o
test_precompiled = $(precompiled) $(object_files_folder)/tests.o



# --------------------- precompilations --------------------- #

$(object_files_folder)/ENVI_reader.o: $(cpp_folder)/ENVI_reader.cpp $(include_folder)/ENVI_reader.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/Results_writer.o: $(cpp_folder)/Results_writer.cpp $(include_folder)/Results_writer.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/Analyzer_tools.o: $(cpp_folder)/Analyzer_tools.cpp $(include_folder)/Analyzer_tools.hpp $(include_folder)/ENVI_reader.hpp $(include_folder)/Functors.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/tests.o: $(test_folder)/tests.cpp $(include_folder)/Analyzer_tools.hpp $(include_folder)/ENVI_reader.hpp $(include_folder)/Functors.hpp $(include_folder)/Results_writer.hpp | $(object_files_folder)
	$(cxx) $(include_test) -c $< -o $@



# ----------------------- compilations ---------------------- #

$(test_binary): $(test_precompiled) 
	$(MAKE) clean
	$(cxx) $^ -o $@


# --------------------------- run --------------------------- #

run_test: $(test_binary)
	./$(test_binary) -s $(test_folder)$(test_spectrums) -i $(test_folder)$(JR_folder)$(test_folder_extension)



# -------------------------- utils -------------------------- #

$(object_files_folder):
	mkdir -p $(object_files_folder)

clean:
	rm -f $(binary)
	rm -f $(test_binary)
	rm -f *.txt

clean_all: clean
	rm -rf $(object_files_folder)