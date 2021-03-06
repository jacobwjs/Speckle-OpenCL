
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include <cstdlib>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#include <algorithm>
#include <ctime>

using namespace std;

//The OpenCL C++ bindings, with exceptions
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>


const std::string separator = "---------------------------------------------------\n";


// Use a static data size for simplicity
//
#define X_PIXELS 64
#define Y_PIXELS 64
// We want the number of threads to equal the number of pixls in the CCD.
#define MAX_WORK_ITEMS (X_PIXELS * Y_PIXELS)
#define NUM_EVENTS 1    // Number of times the kernel is executed.  Need to launch same kernel multiple times
// due to timeout of watchdog timer.

#define NUM_PHOTONS 10000


// Structs that hold data that is passed to the GPU.
//
typedef struct Photon {
    float dirx, diry, dirz;     // direction of the photon.
    float x, y, z;              // exit coordinates of the photon.
    float optical_path_length;  // the optical path length.
    float weight;               // it's exit weight.
    float wavelength;           // The 'wavelength' of the photon.
} Photon;


typedef struct Exit_Photons {
    int num_exit_photons;
    Photon p[NUM_PHOTONS];
} Exit_Photons;


typedef struct CCD {
    float x_center, y_center, z;  // location of the CCD in 3-D space.
    float dx; // pixel size (x-axis)
    float dy; // pixel size (y-axis)
    int total_x_pixels, total_y_pixels;     // number of pixels in the x and y-axis.
    int total_pix;

} CCD;

typedef struct Speckle_Image {
    int num_x;
    int num_y;
    float data[X_PIXELS][Y_PIXELS];
} Speckle_Image;


/// Helper function to print kernel build error.
/// ------------------------------------------------
const char* oclErrorString(cl_int error);


/// Responsible for loading data from files in an informative and structured way.
/// -----------------------------------------------------------------------------
/// Holds the filename and timestamp of the file.
/// Used for sorting and loading data.
struct filename_tstamp
{
    std::string filename;
    size_t tstamp;
};
bool SortFunction (struct filename_tstamp a, struct filename_tstamp b) { return (a.tstamp < b.tstamp); }
void printFiles(std::vector<filename_tstamp> files);
int  Get_num_detected_photons(std::string &filename);
void Load_detected_photons_from_file(void);



int main()
{
    /// Get the platforms and devices available on the system.
    /// ------------------------------------------------------------------------
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;

    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0)
    {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << " Global memory size: " << default_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << "\n";
    //std::cout << " Max workgroup size in each dimension: " << default_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>() << "\n";
    std::cout << " Max workgroup size in total: " << default_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
    std::cout << " Number of compute units: " << default_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
    std::cout << " Max clock frequency: " << default_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                          (cl_context_properties)(all_platforms[0])(),
                                          0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);
    all_devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Program::Sources sources;

    cl_int err;
    try {
        cl::CommandQueue queue(context, default_device, 0, &err);
    }
    catch (cl::Error er) {
        std::cerr << "ERROR: " << er.what() << " " << er.err() << "\n";
        exit(EXIT_FAILURE);
    }



    /// Load OpenCL program
    /// ------------------------------------------------------------------------
    cl::Program program;
    //std::string     filename = "speckle-image.opencl";
    std::string     filename = "speckle-image-v2.opencl";
    std::ifstream   kernel_source_file(filename);
    std::string     kernel_prog(std::istreambuf_iterator<char>(kernel_source_file),
                            (std::istreambuf_iterator<char>()));
    try {
        cl::Program::Sources source(1, std::make_pair(kernel_prog.c_str(),
                                                      kernel_prog.length()+1));
        program = cl::Program(context, source);
    }
    catch (cl::Error er) {
        std::cerr << "ERROR: " << er.what() << " " << oclErrorString(er.err()) << "\n";
        exit(EXIT_FAILURE);
    }
    cout << "\nLoaded kernel source file:   " << filename << "\n";



    /// Build OpenCL program
    /// ------------------------------------------------------------------------
    try {
        err = program.build(all_devices, "");
    }
    catch (cl::Error er) {
        std::cerr << "ERROR: " << er.what() << " " << oclErrorString(er.err()) << "\n";
        exit(EXIT_FAILURE);
    }
    cout << "Successfully built program\n";
    //cout << " Build status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(default_device) << "\n";
    //cout << " Build options: " << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(default_device) << "\n";
    //cout << " Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";



    /// Initialize the kernel
    /// ------------------------------------------------------------------------
    cl::Kernel kernel;
    try {
        kernel = cl::Kernel(program, "Speckle", &err);
    }
    catch (cl::Error er) {
        std::cerr << "ERROR: " << er.what() << " " << er.err() << "\n";
    }



    /// Create CPU memory.
    /// ------------------------------------------------------------------------
    // Create the 2D array for the speckle image.
    Speckle_Image *speckle_image = (Speckle_Image *)malloc(sizeof(Speckle_Image));
    speckle_image->num_x = X_PIXELS;
    speckle_image->num_y = Y_PIXELS;
    //speckle_image->data = (float *)malloc(sizeof(float)*X_PIXELS*Y_PIXELS);
    //memset(speckle_image->data, 0.0f, sizeof(float)*X_PIXELS*Y_PIXELS);
    cout << "\n\nData before kernel execution\n\n";
    for (size_t i = 0; i < X_PIXELS; i++)
    {
        for (size_t j = 0; j < Y_PIXELS; j++)
        {
            speckle_image->data[i][j] = 0.0f;
            cout << speckle_image->data[i][j] << ", ";
        }
        cout << "\n";
    }
    // Create exit photons struct that is handed off to the kernel.
    //
    Exit_Photons *photons = (Exit_Photons *)malloc(sizeof(Exit_Photons));
    photons->num_exit_photons = NUM_PHOTONS;
    int cnt;
    for (cnt = 0; cnt < NUM_PHOTONS; cnt++)
    {
        //photons->p[cnt] = (Photon *)malloc(sizeof(Photon));
        photons->p[cnt].x = ((float) rand() / RAND_MAX);
        photons->p[cnt].y = ((float) rand() / RAND_MAX);
        //photons->p[cnt].x = 0.5f;
        //photons->p[cnt].y = 0.5f;
        photons->p[cnt].z = 5.50f;
        photons->p[cnt].weight = ((float) rand() / RAND_MAX);
        photons->p[cnt].wavelength = 532e-9f;
        photons->p[cnt].optical_path_length = ((float) rand() / RAND_MAX)+1;
    }

    // Define the attributes of the CCD.
    //
    CCD *camera = (CCD *)malloc(sizeof(CCD));
    camera->dx = 6.00e-5;
    camera->dy = camera->dx;
    camera->total_x_pixels = X_PIXELS;
    camera->total_y_pixels = Y_PIXELS;
    camera->total_pix = X_PIXELS * Y_PIXELS;
    camera->x_center = 0.5f;
    camera->y_center = 0.5f;
    camera->z = 100.0f;




    /// Create Device memory.
    /// ------------------------------------------------------------------------
    cl::Buffer cl_CCD;             // device memory used for the CCD
    cl::Buffer cl_photons;         // device memory used for the detected photons
    cl::Buffer cl_speckle_image;   // device memory used for the speckle formation

    cl_CCD           = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(CCD), NULL, &err);
    cl_photons       = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Exit_Photons), NULL, &err);
    cl_speckle_image = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Speckle_Image), NULL, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR: Could not allocate device memory\n";
        exit(EXIT_FAILURE);
    }

    /// Push data to device.
    /// ------------------------------------------------------------------------
    cout << "\nPushing data to GPU...\n";
    cl::Event event;
    cl::CommandQueue queue;
    try {
        queue = cl::CommandQueue(context, default_device, 0, &err);
    }
    catch (cl::Error er)
    {
        std::cerr << "ERROR: " << er.what() << er.err() << "\n";
    }

    err = queue.enqueueWriteBuffer(cl_photons, CL_TRUE, 0, sizeof(Exit_Photons),
                                   (void *)photons, 0, &event);
    err |= queue.enqueueWriteBuffer(cl_CCD, CL_TRUE, 0, sizeof(CCD),
                                   (void *)camera, 0, &event);
    err |= queue.enqueueWriteBuffer(cl_speckle_image, CL_TRUE, 0, sizeof(Speckle_Image),
                                   (void *)speckle_image, 0, &event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR: Failed to enqueue data\n";
        exit(-1);
    }

    err =  kernel.setArg(0, cl_speckle_image);
    err |= kernel.setArg(1, cl_photons);
    err |= kernel.setArg(2, cl_CCD);
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR: Failed to set kernel arguments\n";
        exit(EXIT_FAILURE);
    }


    /// Wait for the command queue to finish these commands before proceeding.
    /// ------------------------------------------------------------------------
    queue.finish();


    /// Run the kernel on the device.
    /// ------------------------------------------------------------------------
    cout << "Executing kernel...\n";
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(64,64), cl::NDRange(1,1), NULL, &event);
    queue.finish();


    /// Read result from device to host.
    /// ------------------------------------------------------------------------
    err = queue.enqueueReadBuffer(cl_speckle_image, CL_TRUE, 0, sizeof(Speckle_Image),
                                   (void *)speckle_image, 0, &event);
    queue.finish();
    event.wait();


    /// Write data out to file for imaging.
    /// ------------------------------------------------------------------------
    std::ofstream speckle_data_stream;
    std::string output_filename = "speckle_test_gpu.dat";
    speckle_data_stream.open(output_filename.c_str());
    if (!speckle_data_stream)
    {
        cout << "!!! ERROR: Output file ('" << output_filename << "') could not be opened !!!\n";
        exit(EXIT_FAILURE);
    }
    // Set the precision and width of the data written to file.
    //speckle_data_stream.width(10);
    //speckle_data_stream.setf(std::ios::showpoint | std::ios::fixed);
    speckle_data_stream.precision(5);

    cout << "\n\nData after kernel execution\n\n";
    for (size_t i = 0; i < X_PIXELS; i++)
    {
        for (size_t j = 0; j < Y_PIXELS; j++)
        {
            speckle_data_stream << speckle_image->data[i][j] << "\t";
            cout << speckle_image->data[i][j] << ", ";
        }
        speckle_data_stream << "\n";
        speckle_data_stream.flush();

        cout << "\n";
    }
    speckle_data_stream.flush();



    return 0;

}


/// Responsible for loading in data (detected photons) from the AO simulation results.
///
void Load_detected_photons_from_file(void)
{
    path p_exit_data = "../AO-KWave-MCBoost/Data";
    path p_speckle_data = "../AO-KWave-MCBoost/Data/Speckles";
    if (is_directory(p_exit_data))
    {
        cout << "Data directory found: " << p_exit_data << '\n';

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
    std::vector<filename_tstamp> files;
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

int Get_num_detected_photons(std::string &filename)
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

// Helper function to get error string
// *********************************************************************
const char* oclErrorString(cl_int error)
{
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "";

}

