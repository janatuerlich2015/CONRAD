package edu.stanford.rsl.tutorial.Jana_ashwini;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import ij.ImageJ;

public class OurPhantom extends edu.stanford.rsl.conrad.data.numeric.Grid2D {
	
	public OurPhantom() {
		super(256,256);
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
	
	public float[] getProjection_parallel(float angle) {
		float[] projection = new float[364];
		for(int x = 0; x < getWidth(); x++) {
			for(int y = 0; y < getHeight(); y++) {
				float s = ((float)x-128) * (float) Math.cos(angle) + ((float)y-128) * (float) Math.sin(angle);
				int first_bucket = (int) Math.floor(s);
				projection[first_bucket + 182] += (1 - (s - first_bucket)) * getPixelValue(x, y);
				projection[first_bucket + 1 + 182] += (s - first_bucket) * getPixelValue(x, y);
			}
		}
		return projection;
	}
	
	public float[] getProjection_fan(float beta) {
		float[] projection = new float[100];
		float D = 200.0f;
		for(int x = 0; x < getWidth(); x++) {
			for(int y = 0; y < getHeight(); y++) {
				if(getPixelValue(x,y) != 0.0f) { 
				
					float x_world = x - 128f;
					float y_world = 128f - y;
					float x_new = (float) (x_world * Math.cos(beta) + y_world * Math.sin(beta));
					float y_new = (float) (- x_world * Math.sin(beta) + y_world * Math.cos(beta));
				
					float gamma = (float) ((Math.atan((x_new)/(D - y_new)))/Math.PI * 180.0f) + 50;
				
					int first_bucket = (int) Math.floor(gamma);
					projection[first_bucket] += (1 - (gamma - first_bucket)) * getPixelValue(x, y);
					projection[first_bucket + 1] += (gamma - first_bucket) * getPixelValue(x, y);
				}
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
					float s = (x-128) * (float) Math.cos(angle) + (y-128) * (float) Math.sin(angle) + 182;
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
//			for(int j = 0; j < columns; j++) {
//				filtered_lines.putPixelValue(j, i, unfiltered_lines[i].getRealAtIndex(j));
//			}
		}
		return filtered_lines;
	}
	
	public Grid2D rebin(Grid2D fanogram) {
		Grid2D sinogram = new Grid2D(364, 180);
		for(int theta = 0; theta < 180; theta++) {
			for(int s = 0; s < 364; s++) {
				float theta_radians = (float) (theta/180.f * Math.PI);
				
				float gamma = (float) (Math.asin((s-182)/200.0f));
				
				float gamma_degrees = (float) (gamma/Math.PI * 180.f);
				float beta = theta_radians - gamma;
				if(beta < 0) beta = (float) (2 * Math.PI + beta);
				float beta_degrees = (float) (beta/Math.PI * 180.f);
				int beta_int = (int) Math.floor(beta_degrees);
				
				int first_bucket = 0;
				int second_bucket = 0;
				if(beta_int < 359) {
					first_bucket = beta_int;
					second_bucket = beta_int + 1;
				} else if(beta_int == 359) {
					first_bucket = beta_int;
					second_bucket = 0;
				} else if(beta_int == 360) {
					first_bucket = 0;
					second_bucket = 1;
				}
				
				if(gamma_degrees + 50 < 99) {
					float result_first = InterpolationOperators.interpolateLinear(fanogram.getSubGrid(first_bucket), gamma_degrees + 50);
					float result_second = InterpolationOperators.interpolateLinear(fanogram.getSubGrid(second_bucket), gamma_degrees + 50);
					sinogram.putPixelValue(s, theta, result_first);
				} else {
					float result_first = fanogram.getSubGrid(first_bucket).getAtIndex(99);
					float result_second = fanogram.getSubGrid(second_bucket).getAtIndex(99);
					sinogram.putPixelValue(s, theta, result_first);
				}
			}
		}
		return sinogram;
	}
	
	
	public static void main(String[] args) {
		OurPhantom phantom = new OurPhantom();
		phantom.drawEllipse(128, 128, 115, 62, 0.5f);
		phantom.drawEllipse(178, 135, 45, 30, 0.3f);
		phantom.drawEllipse(78, 135, 45, 30, 0.3f);
		phantom.drawEllipse(128, 105, 15, 15, 0.7f);
		phantom.drawEllipse(110, 145, 10, 8, 0.1f);
		
		ImageJ image = new ImageJ();
		
		phantom.show("Phantom");
		
		Grid2D sinogram = new Grid2D(364, 180);
		for(int i = 0; i < 180; i++) {
			float angle = ((float)i + 0.33f)/180 * (float)Math.PI;
			float[] projection = phantom.getProjection_parallel(angle);
			for(int j = 0; j < 364; j++) {
				sinogram.putPixelValue(j, i, projection[j]);
			}
		}
		sinogram.show("Sinogram");
		
		
		Grid2D fanogram = new Grid2D(100, 360);
		for(int i = 0; i < 360; i++) {
			
			float angle = ((float)i)/180 * (float)Math.PI;
			float[] projection = phantom.getProjection_fan(angle);
			for(int j = 0; j < 100; j++) {
				fanogram.putPixelValue(j, i, projection[j]);
			}
		}
		fanogram.show("Fanogram");
		
		Grid2D sinogram_from_fanogram = phantom.rebin(fanogram);
		sinogram_from_fanogram.show("Sinogram from Fanogram");
		
//		Grid2D filtered_sinogram_fourier = phantom.getFilteredSinogram_FourierDomain(sinogram);
//		filtered_sinogram_fourier.show();
		
		Grid2D filtered_sinogram_spatial = phantom.getFilteredSinogram_SpatialDomain(sinogram);
		filtered_sinogram_spatial.show("Sinogram - filtered with Ram-Lak");
		
		Grid2D filtered_sinogram_from_fanogram_spatial = phantom.getFilteredSinogram_SpatialDomain(sinogram_from_fanogram);
		filtered_sinogram_from_fanogram_spatial.show("Sinogram from Fanogram - filtered with Ram-Lak");
		
//		Grid2D backprojected_image = phantom.getBackprojection(sinogram);
//		backprojected_image.show();
		
//		Grid2D backprojected_image_fourier = phantom.getBackprojection(filtered_sinogram_fourier);
//		backprojected_image_fourier.show();
		
		Grid2D backprojected_image_spatial = phantom.getBackprojection(filtered_sinogram_spatial);
		backprojected_image_spatial.show("Image - filtered with Ram-Lak");
		
		Grid2D backprojected_image_from_fanogram_spatial = phantom.getBackprojection(filtered_sinogram_from_fanogram_spatial);
		backprojected_image_from_fanogram_spatial.show("Image from Fanogram - filtered with Ram-Lak");
	}
	
}
