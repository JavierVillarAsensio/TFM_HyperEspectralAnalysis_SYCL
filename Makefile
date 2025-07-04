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

run_EUC: compile
	./$(binary) -s $(given_spectrums_folder) -i $(analyzer_folder)$(JR_folder)

run_test: compile_test
	./$(test_binary) -s $(test_folder)$(test_spectrums) -i $(test_folder)$(JR_folder)$(test_folder_extension)

compile: clean
	icps -fsycl -Wno-unqualified-std-cast-call -I$(include_folder) $(analyzer_folder)/*.cpp $(ENVI_folder)/*.cpp -o $(binary)

compile_test:
	icpx -fsycl -Wno-unqualified-std-cast-call -I$(include_folder) $(test_folder)/*.cpp $(ENVI_folder)/*.cpp $(analyzer_folder)/Analyzer_tools.cpp -o $(test_binary)

clean:
	rm -f $(binary)
	rm -f $(test_binary)
	rm -f *.txt