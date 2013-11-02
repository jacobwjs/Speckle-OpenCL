

typedef struct Photon {
    float dirx, diry, dirz;     // direction of the photon.
    float x, y, z;              // exit coordinates of the photon.
    float optical_path_length;  // the optical path length.
    float weight;               // it's exit weight.
    float wavelength;           // The 'wavelength' of the photon.
} Photon;

typedef struct Exit_Photons {
    int num_exit_photons;
    Photon p[30000];
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
    float data[128][16];
} Speckle_Image;



float2 exp_alpha(float alpha);

float2 exp_alpha(float alpha)
{

    float cs,sn;

    sn = sincos(alpha,&cs);
    cs = native_cos(alpha);

    return (float2)(cs,sn);

}


__kernel void Speckle(__global struct Speckle_Image *speckle_image,
                      __global struct Exit_Photons *photons,
                      __global struct CCD *ccd)
{
    const float Pi = 3.14159265358979;

    // Calculate the upper left coordinate of the CCD camera.
    //
    float start_x = ccd->x_center - (ccd->total_x_pixels/2)*ccd->dx;
    float start_y = ccd->y_center - (ccd->total_y_pixels/2)*ccd->dy;

    // Holds the value for the current cartesian coord of the corresponding pixel.
    //
    float x_pixel = 0.0f;
    float y_pixel = 0.0f;

    // The final optical path length (medium + distance to CCD).
    //
    float L = 0.0f;

    // Holds the final intensity calculated at a given pixel. It is updated for
    // each photon.
    float intensity = 0.0f;

    for (int n = 0; n < 30000; n++)
    {
        int i = get_global_id(0);
        int j = get_global_id(1);

        // Calculate the x,y location in cartesian space of this pixel.
        //
        x_pixel = start_x + (ccd->dx * (i+1));
        y_pixel = start_y + (ccd->dy * (j+1));


        // Calculate the distance from the exit aperture of the medium to the pixel
        // on the camera.
        //
        float dist_to_pixel = rsqrt((ccd->z*ccd->z) +
                                    (x_pixel - photons->p[i].x) * (x_pixel - photons->p[i].x) +
                                    (y_pixel - photons->p[i].y) * (y_pixel - photons->p[i].y));

        // Finish the calculation of the total optical path length to the CCD pixel.
        // NOTE: Because the final propagation to the medium is in air, we assume
        //       a refractive index of 1.0.
        L = photons->p[i].optical_path_length + (dist_to_pixel*1.0);


        // The complex field calculation.
        //
        float2 temp = (1/dist_to_pixel) * rsqrt(photons->p[i].weight) * exp_alpha((float)(-1*2*L*Pi*1/(photons->p[i].wavelength)));
        float2 complex = temp;


        // Convert from complex field to intensity.
        //
        intensity = rsqrt(complex.x*complex.x + complex.y*complex.y);


        // Update the pixel with the addition of the new intensity contribution.
        //
        speckle_image->data[i][j] += intensity;

        //speckle_image->data[i][j] = y_pixel;
        //speckle_image->data[i][j] += photons->p[n].optical_path_length;
    }




    //speckle_image->data[i][j] = j;
    //speckle_image->data[i][15] = j;
};


