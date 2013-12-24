#include "exit_data.h"
#include <cstdlib>
#include <sys/stat.h>
#include <cassert>
#include <sstream>
#include <iostream>
using std::cout;


const std::string separator = "---------------------------------------------------\n";


/// Used to sort filenames based on time stamp.
bool SortFunction (struct filename_tstamp a, struct filename_tstamp b) { return (a.tstamp < b.tstamp); }


ExitData::ExitData(const int num_detected_photons)
{
	values.reserve(num_detected_photons);
}




ExitData::~ExitData()
{

}



/// Load all the data filenames from the AO sim.
void ExitData::Load_and_sort_filenames(std::string &directory)
{
    path p_exit_data = directory;
    path p_speckle_data = "Testing-Speckles";
    if (is_directory(p_exit_data))
    {
        cout << "\nData directory found: " << p_exit_data << '\n';

        /// Check if we have a directory to store the generated speckle data.
        if (is_directory(p_speckle_data))
        {
            cout << "Storing speckle data: " << p_speckle_data << '\n';
        }
        else
        {
            cout << "!!!ERROR: Directory for storing speckle data to location [" << p_speckle_data << "] does not exist.\n";
            exit(1);
        }

    }
    else
    {
        cout << "!!!ERROR: Data directory does not exist.  Given the following path: " << p_exit_data << '\n';
        exit(1);
    }

    /// Get the timestamps of all the files and add them to the vector for sorting later.
    struct stat st;
    //std::vector<filename_tstamp> files;
    for (directory_iterator itr(p_exit_data); itr!=directory_iterator(); ++itr)
    {
        std::string f = itr->path().string(); // + itr->path().filename().string();
        if (stat(f.c_str(), &st) != 0)
        {
            cout << "!!!ERROR: Unable to read time stamp of " << f << '\n';
            cout << "st.st_mtime = " << st.st_mtime << '\n';
            exit(1);
        }

        /// Ignore the seeds file used for generating exit photons (i.e. through exit aperture) and directories.
        if ((itr->path().filename().string() != "seeds_for_exit.dat") && is_regular_file(itr->path()))
        {
            filename_tstamp temp;
            temp.filename = f;
            temp.tstamp = st.st_mtime;
            files.push_back(temp);
        }
    }

    /// Sort the files based on their timestamp.
    std::sort (files.begin(), files.end(), SortFunction);

    /// The number of exit-data files to read in and operate on.
    const int NUM_FILES = files.size();
    int num_detected_photons = Get_num_detected_photons((files.at(0)).filename);
    cout << separator;
    cout << "Processing " << NUM_FILES << " exit data files.\n";
    cout << "Detected photons: " << num_detected_photons << '\n';
    cout << separator;
}


size_t ExitData::Get_num_detected_photons(std::string &filename)
{

    int i = 0;
    std::string line;


    // Input stream.
    std::ifstream temp_stream;
    temp_stream.open(filename.c_str());

    do
    {

        getline(temp_stream,line);
        if (temp_stream.fail())
        {
            break;
        }
        ++i;
    }
    while (temp_stream.good());

    temp_stream.close();

    return i;
}



// Allocate a 2D vector to hold all of the exit aperture data.
void ExitData::loadExitData(const size_t &timestep)
{
    loadExitData(files.at(timestep).filename);
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
