
kernel void addBy(global float *grid1,  global const float *grid2, int const numElements_width, int const numElements_height  )
{
	int iGID_x = get_global_id(0);
	int iGID_y = get_global_id(1);
	
	if(iGID_x >= numElements_width)	
		return;
		
	if(iGID_y >= numElements_height)	
		return;
	
		
	
	grid1[iGID_y * numElements_width + iGID_x] += grid2[iGID_y * numElements_width + iGID_x]; 

}


kernel void backproject(global float *image,  global const float *sinogram, int const image_width, int const image_height, int const sinogram_width, int const detector_p )
{
	int iGID_x = get_global_id(0);
	int iGID_y = get_global_id(1);
	
	if(iGID_x >= image_width)	
		return;
		
	if(iGID_y >= image_height)	
		return;
	
	float sum = 0.0f;
	for(int a = 0; a < 180; a++) {
		float angle = (float) (((float) a)/180 * M_PI);
		float s = (iGID_x-(image_width/2)) * (float) cos(angle) + (iGID_y-(image_height/2)) * (float) sin(angle) + (detector_p/2);
		int s_int = (int) floor(s);
		sum += (1 - (s - s_int)) * sinogram[a * sinogram_width + s_int];
		sum += (s - s_int) * sinogram[a * sinogram_width + s_int + 1];
	}
	image[iGID_y * image_width + iGID_x] = sum;	
	
}
