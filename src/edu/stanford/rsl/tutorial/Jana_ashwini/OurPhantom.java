package edu.stanford.rsl.tutorial.Jana_ashwini;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import ij.ImageJ;

public class OurPhantom extends edu.stanford.rsl.conrad.data.numeric.Grid2D {
	
	public OurPhantom() {
		super(512, 512);
	}
	
	public void drawEllipse(int mid_x, int mid_y, int width, int height, float color) {
		for(int x = 0; x < getWidth(); x++) {
			for(int y=0;y<getHeight();y++) {
				int x_new = x - mid_x;
				int y_new = y - mid_y;
				if(((float)x_new*x_new)/(width*width) + ((float)y_new*y_new)/(height*height) < 1) 
					putPixelValue(x,y,color);
			}
		}		
	}
	
	public float[] getProjection(float angle) {
		float[] projection = new float[726];
		for(int x = 0; x < getWidth(); x++) {
			for(int y = 0; y < getHeight(); y++) {
				float s = (x-256) * (float) Math.cos(angle) + (y-256) * (float) Math.sin(angle);
				int first_bucket = (int) s;
				projection[first_bucket + 363] += (1 - (s - first_bucket)) * getPixelValue(x, y);
				projection[first_bucket + 1 + 363] += (s - first_bucket) * getPixelValue(x, y);
			}
		}
		return projection;
	}
	
	public static void main(String[] args) {
		OurPhantom phantom = new OurPhantom();
		phantom.drawEllipse(256, 256, 200, 125, 0.5f);
		phantom.drawEllipse(356, 270, 90, 60, 0.3f);
		phantom.drawEllipse(156, 270, 90, 60, 0.3f);
		phantom.drawEllipse(256, 210, 30, 30, 0.7f);
		phantom.drawEllipse(220, 290, 20, 15, 0.1f);
		
		ImageJ image = new ImageJ();
		
		phantom.show();
		
		Grid2D sinogram = new Grid2D(180, 726);
		for(int i = 0; i < 180; i++) {
			float angle = ((float)i)/180 * (float)Math.PI;
			float[] projection = phantom.getProjection(angle);
			for(int j = 0; j < 726; j++) {
				sinogram.putPixelValue(i, j, projection[j]);
			}
		}
		sinogram.show();
	}
	
}
