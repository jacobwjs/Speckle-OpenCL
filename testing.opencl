
#define CONFIG_USE_DOUBLE 1

//#if CONFIG_USE_DOUBLE
//#if defined(cl_khr_fp64)  // Khronos extension available?
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#define DOUBLE_SUPPORT_AVAILABLE
//#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#define DOUBLE_SUPPORT_AVAILABLE
//#endif

//#endif // CONFIG_USE_DOUBLE

#if defined(DOUBLE_SUPPORT_AVAILABLE)

// double
typedef double real_t;
typedef double2 real2_t;
typedef double3 real3_t;
typedef double4 real4_t;
typedef double8 real8_t;
typedef double16 real16_t;
#define PI 3.14159265358979323846
#else
// float
typedef float real_t;
typedef float2 real2_t;
typedef float3 real3_t;
typedef float4 real4_t;
typedef float8 real8_t;
typedef float16 real16_t;
#define PI 3.14159265359f

#endif


#define WAVELENGTH 532e-9
// Structs that hold data that is passed to the GPU.
//
typedef struct Photon {
    real_t dirx, diry, dirz;        // direction of the photon.
    real_t x, y, z;                 // exit coordinates of the photon.
    real_t displaced_OPL;           // the optical path lengths.
    real_t refraction_OPL;
    real_t combined_OPL;
    real_t weight;                  // it's exit weight.
    real_t wavelength;              // The 'wavelength' of the photon.
} Photon;

typedef struct Exit_Photons {
    int num_exit_photons;
    Photon p[20000];
} Exit_Photons;


// Define the attributes of the CCD camera as well as its location.
//
#define PIXEL_SIZE 6.25e-6f
#define X_CENTER 0.02250f
#define Y_CENTER 0.01145f
#define DISTANCE_FROM_MEDIUM 0.5f
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
    real_t data[64][64];
} Speckle_Image;


real2_t exp_alpha(real_t alpha);
// Return cos(alpha) + I*sin(alpha)
real2_t exp_alpha(real_t alpha)
{
    real_t real_part,imag_part;
    real_part = native_cos(alpha);
    imag_part = native_sin(alpha);
    //cs = cos(alpha);
    return (real2_t)(real_part, imag_part);
}


__kernel void Speckle(__global struct Speckle_Image *speckle_image,
                      __global struct Exit_Photons *photons,
                      __global struct CCD *ccd)
{

    //const int DIMS = 128;
    real2_t TEMP[64][64];
    int m;
    int n;
    for (m = 0; m < 64; m++)
    {
        for ( n = 0; n < 64; n++)
        {
            TEMP[m][n] = (real2_t)(0,0);
        }
    }

    // Assign the attributes of the CCD to the struct for computation below.
    //
    ccd->x_center = X_CENTER;
    ccd->y_center = Y_CENTER;
    ccd->z = DISTANCE_FROM_MEDIUM;
    ccd->dx = PIXEL_SIZE;
    ccd->dy = PIXEL_SIZE;


    // Calculate the upper left coordinate of the CCD camera.
    //
    real_t start_x = ccd->x_center - (ccd->total_x_pixels/2)*ccd->dx;
    real_t start_y = ccd->y_center - (ccd->total_y_pixels/2)*ccd->dy;


    // Holds the value for the current cartesian coord of the corresponding pixel.
    //
    real_t x_pixel = 0.0f;
    real_t y_pixel = 0.0f;

    // Holds the exit weight of the photon bundle upon leaving the medium.
    //
    real_t weight = 0.0f;

    // The final optical path length (medium + distance to CCD).
    //
    real_t L = 0.0f;

    // Holds the final intensity calculated at a given pixel. It is updated for
    // each photon.
    real_t intensity = 0.0f;

    int i = get_global_id(0);
    int j = get_global_id(1);


    //for (int n = 0; n < photons->num_exit_photons; n++)
    for (int n = 0; n < 200; n++)
    {



        // Assign the exit weight of this photon.
        //
        //weight = photons

        // Calculate the x,y location in cartesian space of this pixel.
        //
        x_pixel = start_x + (ccd->dx * (i+1));
        y_pixel = start_y + (ccd->dy * (j+1));


        // Calculate the distance from the exit aperture of the medium to the pixel
        // on the camera.
        //
        real_t dist_to_pixel = sqrt((ccd->z*ccd->z) +
                                    pow((x_pixel - photons->p[n].x), 2) +
                                    pow((y_pixel - photons->p[n].y), 2));

        // Finish the calculation of the total optical path length to the CCD pixel.
        // NOTE: Because the final propagation to the medium is in air, we assume
        //       a refractive index of 1.0.
        L = photons->p[n].refraction_OPL + (dist_to_pixel*1.0);

        // The complex field calculation.
        //
        real2_t temp = (1/(dist_to_pixel*dist_to_pixel)) * sqrt(photons->p[n].weight) * exp_alpha((real_t)(-1*2*PI*L/WAVELENGTH));
        real2_t complex = exp_alpha((real_t)(-1*2*PI*L));

        // Convert from complex field to intensity.
        //
        intensity = sqrt((complex.x*complex.x + complex.y*complex.y));


        TEMP[i][j] += temp;

        // Update the pixel with the addition of the new intensity contribution.
        //
//        speckle_image->data[i][0] += complex.x;
//        speckle_image->data[i][1] += complex.y;
//        speckle_image->data[i][2+n] = L;

    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    real2_t c = TEMP[i][j];
    speckle_image->data[i][j] = fabs(pow(sqrt(c.x*c.x + c.y*c.y),2));
};





