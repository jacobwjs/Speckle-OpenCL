

typedef struct Photon {
    float dirx, diry, dirz;     // direction of the photon.
    float x, y, z;              // exit coordinates of the photon.
    float optical_path_length;  // the optical path length.
    float weight;               // it's exit weight.
    float wavelength;           // The 'wavelength' of the photon.
} Photon;


typedef struct Exit_Photons {
    int num_exit_photons;
    Photon p[30];
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
    float *data;
} Speckle_Image;

// Return cos(alpha)+I*sin(alpha)
float2 exp_alpha(float alpha);

float2 exp_alpha(float alpha)
{
    
    float cs,sn;
    
    sn = sincos(alpha,&cs);
    cs = native_cos(alpha);
    
    return (float2)(cs,sn);
    
}

//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable;

//==============================================================
// Main execution kernel.
//-----------------------

__kernel void Speckle(__global float *speckle_grid,
                      __global struct Exit_Photons *photons,
                      __global struct CCD *ccd)

{
    
    int CNT = photons->num_exit_photons;
    for (int i = 0; i < CNT; i++)
    {

        
        
        const float Pi = 3.14159265358979323;
        
        // Get the global work item.
        //
        int gid = get_global_id(0);
        
        // If all pixels have been calculated, return.
        //
        if (gid > (ccd->total_pix)) return;
        
        // Calculate the x and y pixel this work item is responsible for.
        //
        float start_x = ccd->x_center - (ccd->total_x_pixels/2)*ccd->dx;
        float start_y = ccd->y_center - (ccd->total_y_pixels/2)*ccd->dy;

        // For each work item, we need to find the x and y-coordinate value of the
        // pixel that this work item is responsible for.
        //
        float x_pixel = start_x + ccd->dx * (gid%ccd->total_x_pixels);
        float y_pixel = start_y + ccd->dy*(gid * 1/ccd->total_y_pixels);

        // Calculate the distance from the exit aperture of the medium to the pixel
        // on the camera.
        //
        float D = 4.0f;
        //    float dist_to_pixel = rsqrt(D*D + (x_pixel - p->x)*(x_pixel - p->x) + (y_pixel - p->y)*(y_pixel - p->y));
        float dist_to_pixel = rsqrt(D*D + (x_pixel - photons->p[i].x) *
                                          (x_pixel - photons->p[i].x) +
                                          (y_pixel - photons->p[i].y)*
                                          (y_pixel - photons->p[i].y));
        float L = photons->p[i].optical_path_length + dist_to_pixel;

        // The calculation.
        //
        float2 temp = (1/dist_to_pixel) * (photons->p[i].weight) * exp_alpha((float)(-1*2*L*Pi*1/(photons->p[i].wavelength)));
        float2 complex = temp;

        // Convert from complex number to intensity.
        //
        float intensity = rsqrt(complex.x*complex.x + complex.y*complex.y);
        //intensity = intensity*intensity;



        // Index into the 1-D array as if it were 2-D.
        //
//        speckle_grid[gid] = speckle_grid[gid] + intensity;
        speckle_grid[gid] += intensity;
    }

};




