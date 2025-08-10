binary = Analyzer
test_binary = Test

include_folder = ./code/include
analyzer_folder = ./code/Analyzer
JR_folder = /jasperRidge2_R198
test_folder_extension = _test
ENVI_folder = ./code/ENVI_reader
test_folder = ./code/test
test_spectrums = /spectrums_test
given_spectrums_folder = ./spectrums/given_spectrums
object_files_folder = ./code/object_files

cxx = icpx -std=c++20 -fsycl -I$(include_folder)
precompiled = $(object_files_folder)/ENVI_reader.o $(object_files_folder)/Analyzer_tools.o
test_precompiled = $(precompiled) $(object_files_folder)/tests.o

# --------------------- precompilations --------------------- #

$(object_files_folder)/ENVI_reader.o: $(ENVI_folder)/ENVI_reader.cpp $(include_folder)/ENVI_reader.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/Analyzer_tools.o: $(analyzer_folder)/Analyzer_tools.cpp $(include_folder)/Analyzer_tools.hpp $(include_folder)/ENVI_reader.hpp $(include_folder)/Functors.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@

$(object_files_folder)/tests.o: $(test_folder)/tests.cpp $(include_folder)/Analyzer_tools.hpp $(include_folder)/ENVI_reader.hpp $(include_folder)/Functors.hpp | $(object_files_folder)
	$(cxx) -c $< -o $@



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