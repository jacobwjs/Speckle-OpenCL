#ifndef EXIT_DATA_H
#define EXIT_DATA_H

#include <vector>
#include <string>
#include <fstream>
using std::ifstream;

class ExitData {
public:
	ExitData(const int num_detected_photons);
	~ExitData();

    
	
	// Load in the data from the exit file.
	void loadExitData(const std::string &filename);


   	std::vector<std::vector<double> > values;
private:
	//std::vector<std::vector<double> > values;

	// Input stream.
	std::ifstream exit_file_stream;
};

#endif //EXIT_DATA_H  
