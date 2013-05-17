//
// File:       hello.c
//
// Abstract:   CCD Grid simulation under GPU control.
////////////////////////////////////////////////////////////////////////////////
#include <assert.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define X_PIXELS 256
#define Y_PIXELS 256
// We want the number of threads to equal the number of pixls in the CCD.
#define MAX_WORK_ITEMS (X_PIXELS * Y_PIXELS)
#define NUM_EVENTS 1    // Number of times the kernel is executed.  Need to launch same kernel multiple times
                        // due to timeout of watchdog timer.

#define NUM_PHOTONS 10000

char * load_program_source(const char *filename);



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





int main(int argc, char** argv)
{
    cl_int err;                         // error code returned from api calls
    
    float results[MAX_WORK_ITEMS];    // results returned from device


    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem CCD_mem;                       // device memory used for the input array
    cl_mem photon_mem;                      // device memory used for the output array
    cl_mem speckle_grid;                // device memory used for the speckle formation
    
    const int EXEC_KERNEL_NUM_TIMES = 2;
    cl_event event[EXEC_KERNEL_NUM_TIMES];                     // Used in profiling
    
    
    
    
    // Initialize results to zero.
    int i;
    for (i = 0; i < MAX_WORK_ITEMS; i++)
        results[i] = 0;
    
    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // Create the compute program from the source buffer
    //
    const char *filename = "../../speckle-image.opencl";
    char *program_source = load_program_source(filename);
    program = clCreateProgramWithSource(context, 1, (const char **)&program_source, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
   
    
    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "Speckle", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    
    // Create Photon struct that is handed off to the kernel.
    //
    Photon *p = (Photon *)malloc(sizeof(Photon));
    p->x = 0.4123f;
    p->y = 0.3039f;
    p->z = 1.0f;
    p->weight = 1;
    p->wavelength = 532e-9;
    p->optical_path_length = 2.34;

    
    
    // Create exit photons struct that is handed off to the kernel.
    //
    Exit_Photons *photons = (Exit_Photons *)malloc(sizeof(Exit_Photons));
    photons->num_exit_photons = NUM_PHOTONS;
    int cnt;
    for (cnt = 0; cnt < NUM_PHOTONS; cnt++)
    {
//        photons->p[cnt] = (Photon *)malloc(sizeof(Photon));
        photons->p[cnt].x = 0.4123f; //rand();
        photons->p[cnt].y = 0.3039f; //rand();
        photons->p[cnt].z = 1.0f;
        photons->p[cnt].weight = 1.0f;
        photons->p[cnt].wavelength = 532e-9f;
        photons->p[cnt].optical_path_length = 1.0;
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
    camera->z = 1.0f;
    
    
    
    // Create the device memory.  No need to write anything to device.
    //
    speckle_grid = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_WORK_ITEMS, NULL, NULL);
    //photon_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Photon), NULL, NULL);
    photon_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Exit_Photons), NULL, NULL);
    CCD_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(CCD), NULL, NULL);
    
    // Ensure allocated properly.
    //
    if (!speckle_grid || !photon_mem || !CCD_mem)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    } 
    
    // Write memory to GPU.
//    err = clEnqueueWriteBuffer(commands, photon_mem, CL_TRUE, 0, sizeof(Photon),
//                                (void *)p, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, photon_mem, CL_TRUE, 0, sizeof(Exit_Photons),
                               (void *)photons, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, CCD_mem, CL_TRUE, 0, sizeof(CCD),
                                (void *)camera, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    
    // Get all of the stuff written and allocated 
    clFinish(commands);
    
    
    // Set the arguments to our compute kernel
    //
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem),  &speckle_grid);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &photon_mem);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &CCD_mem);
    //err |= clSetKernelArg(kernel, 1, sizeof(float), &photon_weight);
    //err |= clSetKernelArg(kernel, 2, sizeof(float), &optical_path_length);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    
    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    
    //printf("local work-group-size = %d\n", (int)local);
    
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = MAX_WORK_ITEMS;
    //local = 128;
    //local = 1;
    //local = MAX_THREADS / 2;
    
    
    //err  = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &event[0]);
    
    for (cnt = 0; cnt < EXEC_KERNEL_NUM_TIMES; ++cnt)
    {
        //err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &event[num_exit_photons]);
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, &event[cnt]);
        //err |= clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, &event[1]);
        if (err)
        {
            printf("Error: Failed to execute kernel!\n");
            return EXIT_FAILURE;
        }
    
        //clFinish(commands);
        //clWaitForEvents(num_exit_photons, event);


    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);
    //clWaitForEvents(2, event);

    
    
    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, speckle_grid, CL_TRUE, 0, sizeof(float) * MAX_WORK_ITEMS, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);
    
    // For profiling execution time.
    cl_ulong start, end;
    clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event[cnt-1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    //clGetEventProfilingInfo(event[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    //printf("Time for event (ms): %10.5f  \n", (end-start)/1000000.0);
   
                                
    // Print out the calculated fluences to disk.
    //printVals();
    //for (i = 0; i < MAX_WORK_ITEMS; i++)
    //    printf("results[%i] = %f\n", i, results[i]);
    
    
    printf("Time for event (ms): %10.5f  \n", (end-start)/1000000.0);

    
    // Shutdown and cleanup
    //
    if (p) 
        free(p);
    
    if (camera)
        free(camera);

    clReleaseMemObject(speckle_grid);
    clReleaseMemObject(photon_mem);
    clReleaseMemObject(CCD_mem);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    for (i = 0; i < EXEC_KERNEL_NUM_TIMES; i++)
        clReleaseEvent(event[i]);
    
    return 0;
}


char * load_program_source(const char *filename)
{
    struct stat statbuf;
    FILE *fp;
    char *source;
    fp = fopen(filename, "r");
    assert (fp != NULL);
    stat(filename, &statbuf);
    source = (char *)malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fp);
    source[statbuf.st_size] = '\0';
    
    //printf("kernel = %s", source);
    
    return source;
}







