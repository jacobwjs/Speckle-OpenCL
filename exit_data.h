#ifndef EXIT_DATA_H
#define EXIT_DATA_H


#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#include <vector>
#include <string>
#include <fstream>
using std::ifstream;


/// Holds the filename and timestamp of the file.
/// Used for sorting and loading data.
struct filename_tstamp
{
    std::string filename;
    size_t tstamp;
};


class ExitData {
public:
	ExitData(const int num_detected_photons);
	~ExitData();

    
    /// Parse the directory where the detected photon data files reside and
    /// create a vector that holds all of the names for later loading in the data
    /// for forming a speckle pattern at a given time step in the AO sim.
    void Load_and_sort_filenames(std::string &directory);

	// Load in the data from the exit file.
	void loadExitData(const std::string &filename);
    void loadExitData(const size_t &timestep);

    /// Returns how many photons were detected and recorded in this data file.
    /// NOTE:
    ///  - Each data file represents a time step in the AO simulation.
    size_t Get_num_detected_photons(std::string &filename);

    /// Returns the number of files in the data directory.
    size_t Get_num_files() {return files.size();}


   	std::vector<std::vector<double> > values;


private:
	//std::vector<std::vector<double> > values;
    std::vector<filename_tstamp> files;

	// Input stream.
	std::ifstream exit_file_stream;
};

#endif //EXIT_DATA_H  
