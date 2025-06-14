binary = Analyzer
test_binary = Test
include_folder = ./code/include
analyzer_folder = ./code/Analyzer
ENVI_folder = ./code/ENVI_reader
test_folder = ./code/test

compile:
	g++ -I$(include_folder) $(analyzer_folder)/*.cpp $(ENVI_folder)/*.cpp -o $(binary)

compile_test:
	g++ -I$(include_folder) $(test_folder)/*.cpp $(ENVI_folder)/*.cpp -o $(test_binary)

clean:
	rm -f $(binary)
	rm -f $(test_binary)