
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <cstdlib>
//#include <sys/stat.h>
//#include <boost/filesystem.hpp>
//using namespace boost::filesystem;
#include <algorithm>
#include <ctime>
#include <sstream>      // std::stringstream
using namespace std;

#include "exit_data.h"

//The OpenCL C++ bindings, with exceptions
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>




#define DEBUG
#undef DEBUG




//#if defined (cl_khr_fp64)
//    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#elif defined(cl_amd_fp64)
//    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#else
//    #define NO_DOUBLE_SUPPORT 1
//#endif






// We want the number of threads to equal the number of pixls in the CCD.
//#define MAX_WORK_ITEMS (X_PIXELS * Y_PIXELS)
//#define NUM_EVENTS 1    // Number of times the kernel is executed.  Need to launch same kernel multiple times
// due to timeout of watchdog timer.

#define NUM_PHOTONS 25000
//#define NUM_PHOTONS 320000



//#define DOUBLE_SUPPORT_AVAILABLE

#if defined(DOUBLE_SUPPORT_AVAILABLE)
// double
typedef double real_t;
typedef cl_double2 real2_t;
#define PI 3.14159265358979323846
#else
// float
typedef float real_t;
typedef cl_float2 real2_t;
#define PI 3.14159265359f
#endif

// Structs that hold data that is passed to the GPU.
//
typedef struct Photon {
    real_t dirx, diry, dirz;        // direction of the photon.
    real_t x, y, z;                 // exit coordinates of the photon.
    real_t displaced_OPL;           // the optical path length.
    real_t refraction_OPL;
    real_t combined_OPL;
    real_t weight;                  // it's exit weight.
    real_t wavelength;              // The 'wavelength' of the photon.
} Photon;


typedef struct Exit_Photons {
    int num_exit_photons;
    Photon p[NUM_PHOTONS];
} Exit_Photons;


// Define the attributes of the CCD camera as well as its location.
//
#define X_PIXELS 512
#define Y_PIXELS 512
typedef struct CCD {
    real_t x_center, y_center, z;  // location of the CCD in 3-D space.
    real_t dx; // pixel size (x-axis)
    real_t dy; // pixel size (y-axis)
    int total_x_pixels, total_y_pixels;     // number of pixels in the x and y-axis.
    int total_pix;
} CCD;



typedef struct Speckle_Image {
    int num_x;
    int num_y;
    real_t data[X_PIXELS][Y_PIXELS];
    real2_t temp_data[X_PIXELS][Y_PIXELS];
} Speckle_Image;


/// Helper function to print kernel build error.
/// ------------------------------------------------
const char* oclErrorString(cl_int error);




int main()
{
    //if (NO_DOUBLE_SUPPORT)  cout << "Double precision not supported\n";

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
    cl::Platform default_platform = all_platforms[0];
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
    //std::string     filename = "speckle-image-v2.opencl";
    std::string     filename = "testing.opencl";
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
    //speckle_image->data = (real_t *)malloc(sizeof(real_t)*X_PIXELS*Y_PIXELS);
    //memset(speckle_image->data, 0.0f, sizeof(real_t)*X_PIXELS*Y_PIXELS);
    cout << "\n\nData before kernel execution\n\n";
    for (size_t i = 0; i < X_PIXELS; i++)
    {
        for (size_t j = 0; j < Y_PIXELS; j++)
        {
            speckle_image->data[i][j] = 0.0f;
            //speckle_image->temp_data[i][j] = {0.0f, 0.0f};
#ifdef DEBUG
            cout << speckle_image->data[i][j] << ", ";
#endif
        }
#ifdef DEBUG
        cout << "\n";
#endif
    }


    /// Create exit photons struct that is handed off to the kernel.
    ///
    Exit_Photons *photons = (Exit_Photons *)malloc(sizeof(Exit_Photons));
    photons->num_exit_photons = NUM_PHOTONS;
    int cnt;

    /// Define the attributes of the CCD.
    ///
    CCD *camera = (CCD *)malloc(sizeof(CCD));
    camera->total_x_pixels = X_PIXELS;
    camera->total_y_pixels = Y_PIXELS;
    camera->total_pix = X_PIXELS * Y_PIXELS;


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




    /// Process the detected photon data files for each time step in the AO simulation
    /// and form the speckle patterns that will be used to create AO signal curve.
    /// --------------------------------------------------------------------------------------------

    /// Create the exit data object that holds the detected photon data.
    ///
    ExitData * detected_photons = new ExitData(NUM_PHOTONS);
    std::string data_dir_path = "Data/25K_photons";
    detected_photons->Load_and_sort_filenames(data_dir_path);

    /// Holds the values for the exit data during assignment.
    ///
    real_t weight            = 0.0f;
    real_t displaced_OPL     = 0.0f;
    real_t refraction_OPL    = 0.0f;
    real_t combined_OPL      = 0.0f;
    real_t exit_location_x   = 0.0f;
    real_t exit_location_y   = 0.0f;
    real_t exit_location_z   = 0.0f;

    cout << "\nLoading " << NUM_PHOTONS << " detected photons for speckle pattern formation\n";
    /// For each time step of recorded data from the AO sim
    for (size_t time_step = 0; time_step < detected_photons->Get_num_files(); time_step++)
    //for (size_t time_step = 0; time_step < 1; time_step++)
    {
        /// Load in the data for this time step.
        detected_photons->loadExitData(time_step);

        /// Loop through the exit data and assign 'NUM_PHOTONS' of detected photons
        /// the struct that will be handed off to the OpenCL kernel.
        for (cnt = 0; cnt < NUM_PHOTONS; cnt++)
        {

            // Transfer data from the 'ExitData' object to the structs that OpenCL uses.
            //
            weight 				= detected_photons->values[cnt][0];
            displaced_OPL 		= detected_photons->values[cnt][1];
            refraction_OPL      = detected_photons->values[cnt][2];
            combined_OPL		= detected_photons->values[cnt][3];
            exit_location_x 	= detected_photons->values[cnt][4];
            exit_location_y		= detected_photons->values[cnt][5];
            //exit_location_z		= exit_data_all_timesteps.at(time_step)->values[cnt][6];


            photons->p[cnt].x = exit_location_x;
            photons->p[cnt].y = exit_location_y;
            //photons->p[cnt].z = exit_location_z;
            photons->p[cnt].displaced_OPL   = displaced_OPL;
            photons->p[cnt].refraction_OPL  = refraction_OPL;
            photons->p[cnt].combined_OPL    = combined_OPL;
            photons->p[cnt].weight = weight;
            photons->p[cnt].wavelength = 532e-9f;
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

        err = queue.enqueueWriteBuffer(cl_speckle_image, CL_TRUE, 0, sizeof(Speckle_Image),
                                       (void *)speckle_image, 0, &event);
        err |= queue.enqueueWriteBuffer(cl_photons, CL_TRUE, 0, sizeof(Exit_Photons),
                                       (void *)photons, 0, &event);
        err |= queue.enqueueWriteBuffer(cl_CCD, CL_TRUE, 0, sizeof(CCD),
                                       (void *)camera, 0, &event);


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
        err = queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(X_PIXELS,Y_PIXELS), cl::NDRange(16,16), NULL, &event);
        queue.finish();


        /// Read result from device to host.
        /// ------------------------------------------------------------------------
        err = queue.enqueueReadBuffer(cl_speckle_image, CL_TRUE, 0, sizeof(Speckle_Image),
                                       (void *)speckle_image, 0, &event);
        queue.finish();
        event.wait();


        /// Write speckle data out to file for calculation and/or imaging.
        /// ------------------------------------------------------------------------
        std::ofstream speckle_data_stream;
        std::stringstream output_filename;
        output_filename << "Testing-Speckles/speckle_t" << time_step << ".dat";
        ///std::string output_filename = "Testing-Speckles/speckle_gpu.dat";
        speckle_data_stream.open(output_filename.str().c_str());
        if (!speckle_data_stream)
        {
            cout << "!!! ERROR: Output file ('" << output_filename << "') could not be opened !!!\n";
            exit(EXIT_FAILURE);
        }
        // Set the precision and width of the data written to file.
        //speckle_data_stream.width(10);
        //speckle_data_stream.setf(std::ios::showpoint | std::ios::fixed);
        speckle_data_stream.precision(6);

   #ifdef DEBUG
        cout << "\n\nData after kernel execution\n\n";
   #endif
        for (size_t i = 0; i < X_PIXELS; i++)
        {
            for (size_t j = 0; j < Y_PIXELS; j++)
            {
                speckle_data_stream << speckle_image->data[i][j] << "\t";
    #ifdef DEBUG
                cout << speckle_image->data[i][j] << ", ";
    #endif
            }
            speckle_data_stream << "\n";
            speckle_data_stream.flush();
    #ifdef DEBUG
            cout << "\n";
    #endif
        }
        speckle_data_stream.flush();



    } /// End for loop (num_files)






    return 0;

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

