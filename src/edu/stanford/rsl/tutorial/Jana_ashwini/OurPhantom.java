package edu.stanford.rsl.tutorial.Jana_ashwini;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
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
				float s = ((float)x-256.5f) * (float) Math.cos(angle) + ((float)y-256.5f) * (float) Math.sin(angle);
				int first_bucket = (int) Math.floor(s);
				projection[first_bucket + 363] += (1 - (s - first_bucket)) * getPixelValue(x, y);
				projection[first_bucket + 1 + 363] += (s - first_bucket) * getPixelValue(x, y);
			}
		}
		return projection;
	}
	
	public Grid2D getBackprojection(Grid2D sinogram) {
		Grid2D image = new Grid2D(getWidth(), getHeight());
		for(int x = 0; x < getWidth(); x++) {
			for(int y = 0; y < getHeight(); y++) {
				float sum = 0.0f;
				for(int a = 0; a < 180; a++) {
					float angle = (float) (((float) a)/180 * Math.PI);
					float s = (x-256) * (float) Math.cos(angle) + (y-256) * (float) Math.sin(angle) + 363;
					sum += InterpolationOperators.interpolateLinear(sinogram.getSubGrid(a), s);
				}
				image.putPixelValue(x, y, sum);
			}
		}
		return image;
	}
	
	public Grid2D getFilteredSinogram_FourierDomain(Grid2D sinogram) {
		Grid1DComplex[] unfiltered_lines = new Grid1DComplex[sinogram.getHeight()];
		for(int i = 0; i < unfiltered_lines.length; i++) {
			unfiltered_lines[i] = new Grid1DComplex(sinogram.getSubGrid(i));
			unfiltered_lines[i].transformForward(); 
			int n = unfiltered_lines[i].getSize()[0];
			for(int j = 0; j < n/2; j++) {
				unfiltered_lines[i].multiplyAtIndex(j, 0.5f*j);
			}
			for(int j = n/2; j < n; j++) {
				unfiltered_lines[i].multiplyAtIndex(j, 0.5f*(n-j));
			}
			unfiltered_lines[i].transformInverse();
		}
		Grid2D filtered_lines = new Grid2D(sinogram.getWidth(), sinogram.getHeight());
		for(int i = 0; i < filtered_lines.getHeight(); i++) {
			for(int j = 0; j < filtered_lines.getWidth(); j++) {
				filtered_lines.putPixelValue(j, i, unfiltered_lines[i].getAtIndex(j));
			}
		}
		return filtered_lines;
	}
	
	public Grid2D getFilteredSinogram_SpatialDomain(Grid2D sinogram) {
		Grid1DComplex[] unfiltered_lines = new Grid1DComplex[sinogram.getHeight()];
		
		Grid1D ramlak = new Grid1D(sinogram.getWidth());
		for(int i = 0; i < sinogram.getWidth(); i++) {
			int number = i - sinogram.getWidth()/2;
			if(number == 0) {
				ramlak.setAtIndex(i, 0.25f);
			} else if(number % 2 == 0) {
				ramlak.setAtIndex(i, 0);
			} else {
				ramlak.setAtIndex(i, -1.0f/((float)(number*number*Math.PI*Math.PI)));
			}
		}
		Grid1DComplex ramlak_filter = new Grid1DComplex(ramlak);
		ramlak_filter.transformForward();
		
		ramlak_filter.show();
		
		for(int i = 0; i < unfiltered_lines.length; i++) {
			unfiltered_lines[i] = new Grid1DComplex(sinogram.getSubGrid(i));
			unfiltered_lines[i].transformForward(); 
			int n = unfiltered_lines[i].getSize()[0];
			for(int j = 0; j < n; j++) {
				unfiltered_lines[i].multiplyAtIndex(j, ramlak_filter.getRealAtIndex(j));
			}
			unfiltered_lines[i].transformInverse();
		}
		Grid2D filtered_lines = new Grid2D(sinogram.getWidth(), sinogram.getHeight());
		for(int i = 0; i < filtered_lines.getHeight(); i++) {
			int columns = filtered_lines.getWidth();
			for(int j = 0; j < columns/2; j++) {
				filtered_lines.putPixelValue(j, i, unfiltered_lines[i].getRealAtIndex(j + columns/2));
			}
			for(int j = columns/2; j < columns; j++) {
				filtered_lines.putPixelValue(j, i, unfiltered_lines[i].getRealAtIndex(j - columns/2));
			}
		}
		return filtered_lines;
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
		
		Grid2D sinogram = new Grid2D(726, 180);
		for(int i = 0; i < 180; i++) {
			float angle = ((float)i + 0.33f)/180 * (float)Math.PI;
			float[] projection = phantom.getProjection(angle);
			for(int j = 0; j < 726; j++) {
				sinogram.putPixelValue(j, i, projection[j]);
			}
		}
		sinogram.show();
		
		
		
		Grid2D filtered_sinogram_fourier = phantom.getFilteredSinogram_FourierDomain(sinogram);
		filtered_sinogram_fourier.show();
		
		Grid2D filtered_sinogram_spatial = phantom.getFilteredSinogram_SpatialDomain(sinogram);
		filtered_sinogram_spatial.show();
		
		Grid2D backprojected_image = phantom.getBackprojection(sinogram);
		backprojected_image.show();
		
		Grid2D backprojected_image_fourier = phantom.getBackprojection(filtered_sinogram_fourier);
		backprojected_image_fourier.show();
		
		Grid2D backprojected_image_spatial = phantom.getBackprojection(filtered_sinogram_spatial);
		backprojected_image_spatial.show();
	}
	
}
