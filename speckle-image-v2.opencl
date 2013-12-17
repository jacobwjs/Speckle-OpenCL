

//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

typedef cl_double real_t;
typedef float2 real2_t;


typedef struct Photon {
    real_t dirx, diry, dirz;     // direction of the photon.
    real_t x, y, z;              // exit coordinates of the photon.
    real_t optical_path_length;  // the optical path length.
    real_t weight;               // it's exit weight.
    real_t wavelength;           // The 'wavelength' of the photon.
} Photon;

typedef struct Exit_Photons {
    int num_exit_photons;
    Photon p[1000];
} Exit_Photons;


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

real2_t exp_alpha(real_t alpha)
{

    real_t cs,sn;

    sn = sincos(alpha,&cs);
    //cs = native_cos(alpha);

    return (real2_t)(cs,sn);

}


__kernel void Speckle(__global struct Speckle_Image *speckle_image,
                      __global struct Exit_Photons *photons,
                      __global struct CCD *ccd)
{
const real_t Pi = 3.14159265358979;

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

for (int n = 0; n < photons->num_exit_photons; n++)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

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
    real_t dist_to_pixel = rsqrt((ccd->z*ccd->z) +
                                (x_pixel - photons->p[n].x) * (x_pixel - photons->p[n].x) +
                                (y_pixel - photons->p[n].y) * (y_pixel - photons->p[n].y));




    // Finish the calculation of the total optical path length to the CCD pixel.
    // NOTE: Because the final propagation to the medium is in air, we assume
    //       a refractive index of 1.0.
    L = photons->p[n].optical_path_length + (dist_to_pixel*1.0);


    // The complex field calculation.
    //
    real2_t temp = (1/dist_to_pixel) * rsqrt(photons->p[n].weight) * exp_alpha((real_t)(-1*2*L*Pi*1/(photons->p[n].wavelength)));
    real2_t complex = temp;


    // Convert from complex field to intensity.
    //
    intensity = rsqrt(complex.x*complex.x + complex.y*complex.y);


    // Update the pixel with the addition of the new intensity contribution.
    //
    speckle_image->data[i][j] += intensity;

    //speckle_image->data[i][j] = dist_to_pixel;
}




//speckle_image->data[i][j] = j;
//speckle_image->data[i][15] = j;


};


