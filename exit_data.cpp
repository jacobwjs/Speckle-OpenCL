#include "exit_data.h"
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <iostream>
using std::cout;


ExitData::ExitData(const int num_detected_photons)
{
	values.reserve(num_detected_photons);
}


ExitData::~ExitData()
{

}

// Allocate a 2D vector to hold all of the exit aperture data.
void ExitData::loadExitData(const std::string &filename)
{
	
    // Open the file that contains the exit data from the medium's aperture.
	//
	if (exit_file_stream.is_open())
	{
        exit_file_stream.close();
    }
   
	exit_file_stream.open(filename.c_str());
    if (!exit_file_stream)
    {
        cout << "!!! ERROR: Could not open file (" << filename << ")\n";
        exit(1);
    }

	/// To know how large to make each vector we need to know how many data points
	/// were written out to file while collecting data on exit photons.  Data is
	/// written line by line, and based on what is chosen to be saved in the simulation
	/// (e.g. weight, coords, etc.) this value changes.  Here we read in one line and
	/// find out how many data points are on a single line.
	std::istringstream stream1;
	std::string line;
	getline(exit_file_stream, line);
	stream1.str(line);
	double temp_num;
	size_t cnt = 0;	
	while (stream1 >> temp_num) cnt++;

	size_t COLS = cnt;
	cout << "Number of exit data entries per photon: " << COLS << '\n';
	cout.flush();
    

	/// Capacity has been reserved upon creation, but to remove any issues
	/// with old data we clear the vector before we populate it.
	values.clear();
	assert(values.size() == 0);
	

    // Read and store the exit data to the 2D array.
    double temp = 0.0;
	size_t i = 0;
	do
	{	 
	        values.push_back(std::vector<double>(COLS));
	        
	        for (size_t j = 0; j < COLS; j++)
	        {
	            exit_file_stream >> temp;
				if (exit_file_stream.fail())
				{
					break;
				}

	            values[i][j] = temp; 
	        }
 		++i;
	}
	while (exit_file_stream.good());
}
