package edu.stanford.rsl.tutorial.Jana_ashwini;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import ij.ImageJ;

public class OurPhantom extends edu.stanford.rsl.conrad.data.numeric.Grid2D {
	
	private float D;
	private int opening_angle = 100;
	private int detector_p;
	private int ppd = 2;
	private int detector_f;
	
	private int width;
	
	CLContext context;
	CLDevice device;
	CLProgram program;
	int localWorkSize;
	int globalWorkSizeHeight;
	int globalWorkSizeWidth;
	boolean initialized;
	
	public OurPhantom() {
		super(512, 512);
		this.width = getWidth();
		this.detector_p = (int) Math.ceil(Math.sqrt(2) * width);
		if(detector_p % 2 == 1) detector_p++;
		this.D = width;
		this.detector_f = (int) Math.ceil(4 * D * Math.atan(opening_angle/2 * Math.PI / 180));
		initialized = false;
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
	
	public Grid2D getSinogram() {
		Grid2D sinogram = new Grid2D(detector_p, 180);
		for(int i = 0; i < 180; i++) {
			float angle = ((float)i)/180 * (float)Math.PI;
			float[] projection = getProjection_parallel(angle);
			for(int j = 0; j < detector_p; j++) {
				sinogram.putPixelValue(j, i, projection[j]);
			}
		}
		return sinogram;
	}
	
	public float[] getProjection_parallel(float angle) {
		float[] projection = new float[detector_p];
		for(int x = 0; x < getWidth(); x++) {
			for(int y = 0; y < getHeight(); y++) {
				float s = ((float)x-width/2) * (float) Math.cos(angle) + ((float)y-width/2) * (float) Math.sin(angle);
				int first_bucket = (int) Math.floor(s);
				projection[first_bucket + detector_p/2] += (1 - (s - first_bucket)) * getPixelValue(x, y);
				projection[first_bucket + 1 + detector_p/2] += (s - first_bucket) * getPixelValue(x, y);
			}
		}
		return projection;
	}
	
	public Grid2D getFanogram(boolean short_scan, boolean equally_spaced) {
		int degrees_to_go = 360;
		int width = opening_angle*ppd;
		if(short_scan) degrees_to_go = 180 + opening_angle;
		if(equally_spaced) width = detector_f;
		Grid2D fanogram = new Grid2D(width, degrees_to_go);
		for(int i = 0; i < degrees_to_go; i++) {
			
			float angle = ((float)i)/180 * (float)Math.PI;
			float[] projection = getProjection_fan(angle, equally_spaced);
			for(int j = 0; j < width; j++) {
				fanogram.putPixelValue(j, i, projection[j]);
			}
		}
		return fanogram;
	}
	
	public float[] getProjection_fan(float beta, boolean equally_spaced) {
		int width = opening_angle*ppd;
		if(equally_spaced) width = detector_f;
		float[] projection = new float[width];
		for(int x = 0; x < getWidth(); x++) {
			for(int y = 0; y < getHeight(); y++) {
				if(getPixelValue(x,y) != 0.0f) { 
				
					float x_world = x - this.width/2;
					float y_world = this.width/2 - y;
					float x_new = (float) (x_world * Math.cos(beta) + y_world * Math.sin(beta));
					float y_new = (float) (- x_world * Math.sin(beta) + y_world * Math.cos(beta));
				
//					float gamma = (float) (ppd * (Math.atan((x_new)/(D - y_new)))/Math.PI * 180.0f + opening_angle*ppd/2);
					
					float gamma = (float) ((Math.atan(x_new/(D - y_new))));
					float t = (float) (2 * D * Math.tan(gamma)) + detector_f/2;
					
					
//					int first_bucket = (int) Math.floor(gamma);
//					projection[first_bucket] += (1 - (gamma - first_bucket)) * getPixelValue(x, y);
//					projection[first_bucket + 1] += (gamma - first_bucket) * getPixelValue(x, y);
					
					int first_bucket = (int) Math.floor(t);
					projection[first_bucket] += (1 - (t - first_bucket)) * getPixelValue(x, y);
					projection[first_bucket + 1] += (t - first_bucket) * getPixelValue(x, y);
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
					float s = (x-width/2) * (float) Math.cos(angle) + (y-width/2) * (float) Math.sin(angle) + detector_p/2;
					sum += InterpolationOperators.interpolateLinear(sinogram.getSubGrid(a), s);
				}
				image.putPixelValue(x, y, sum);
			}
		}
		return image;
	}
	
	public void initializeCL() {
		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();
		program = null;
		try {
			program = context.createProgram(OurPhantom.class.getResourceAsStream("Addition.cl")).build();
		} catch(Exception e) {
			System.err.println("Could not build program.");
		}
		
		localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16); 
		globalWorkSizeHeight = OpenCLUtil.roundUp(localWorkSize, width); 
		globalWorkSizeWidth = OpenCLUtil.roundUp(localWorkSize, width);
		
	}
	
	public Grid2D getBackprojectionCL(Grid2D sinogram) {
		Grid2D image = new Grid2D(getWidth(), getHeight());
		
		if(!initialized) {
			initializeCL();
			initialized = true;
		}
		
		CLBuffer<FloatBuffer> imageBuffer = context.createFloatBuffer(image.getWidth() * image.getHeight(), Mem.WRITE_ONLY);
		CLBuffer<FloatBuffer> sinogramBuffer = context.createFloatBuffer(sinogram.getWidth() * sinogram.getHeight(), Mem.READ_ONLY);
		
		for (int i = 0; i < image.getNumberOfElements(); ++i){
			imageBuffer.getBuffer().put(image.getBuffer()[i]);
		}
		for (int i = 0; i < sinogram.getNumberOfElements(); ++i){
			sinogramBuffer.getBuffer().put(sinogram.getBuffer()[i]);
		}
		imageBuffer.getBuffer().rewind();
		sinogramBuffer.getBuffer().rewind();
		
		CLKernel kernelFunction = program.createCLKernel("backproject");
		kernelFunction.putArg(imageBuffer).putArg(sinogramBuffer).putArg(image.getWidth()).putArg(image.getHeight()).putArg(sinogram.getWidth()).putArg(detector_p);
		
		CLCommandQueue queue = device.createCommandQueue();
		queue
		.putWriteBuffer(imageBuffer, true)
		.putWriteBuffer(sinogramBuffer, true)
		.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSizeWidth, globalWorkSizeHeight,
				localWorkSize, localWorkSize).finish()
				.putReadBuffer(imageBuffer, true)
				.finish();
		
		for (int i = 0; i < image.getWidth() * image.getHeight(); ++i) {
				image.getBuffer()[i] = imageBuffer.getBuffer().get();
		}
		
		queue.release();
		imageBuffer.release();
		sinogramBuffer.release();
		kernelFunction.release();
//		program.release();
//		context.release();
		
		
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
				filtered_lines.putPixelValue(j, i, unfiltered_lines[i].getRealAtIndex(j));
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
		}
		return filtered_lines;
	}
	
	public Grid2D rebin(Grid2D fanogram, boolean equally_spaced) {
		Grid2D sinogram = new Grid2D(detector_p, 180);
		for(int theta = 0; theta < 180; theta++) {
			for(int s = 0; s < detector_p; s++) {
				
				float theta_radians = (float) -(theta/180.f * Math.PI);
				
				float gamma = (float) (Math.asin((s-detector_p/2)/D));
				
				float t = (float) (2 * D * Math.tan(gamma)) + detector_f/2;
				
				float gamma_degrees = (float) (gamma/Math.PI * 180.f);
				
				float gamma_index = ppd * gamma_degrees + ppd*opening_angle/2;
				
				float beta = theta_radians - gamma;
				if(beta < 0) beta = (float) (2 * Math.PI + beta);
				float beta_degrees = (float) (beta/Math.PI * 180.f);
				int beta_int = (int) Math.floor(beta_degrees);
			
				if(beta_int > fanogram.getHeight() - 1) { //short-scan compatibility
					gamma = -gamma;
//					if(gamma > 2 * Math.PI) gamma -= 2 * Math.PI;
					t = (float) (2 * D * Math.tan(gamma)) + detector_f/2;
					gamma_degrees = (float) (gamma/Math.PI * 180.f);
					gamma_index = ppd * gamma_degrees + ppd*opening_angle/2;
				
					beta = (float) (beta - 2 * gamma - Math.PI);
					//if(beta < 0) beta = (float) (2 * Math.PI + beta);
					beta_degrees = (float) (beta/Math.PI * 180.f);
					beta_int = (int) Math.floor(beta_degrees);
				}
				
				int first_bucket = beta_int;
//				//int second_bucket = 0;
//				if(beta_int < 359) {
//					first_bucket = beta_int;
//					//second_bucket = beta_int + 1;
//				} else if(beta_int == 359) {
//					first_bucket = beta_int;
//					//second_bucket = 0;
//				} else if(beta_int == 360) {
//					first_bucket = 0;
//					//second_bucket = 1;
//				}
				
				if(equally_spaced) {
					if(t < 0) {
						sinogram.putPixelValue(s, theta, 0.0f);
					} else if (t >= detector_f) {
						sinogram.putPixelValue(s, theta, 0.0f);
					} else if(t < detector_f - 1) {
						float result_first = InterpolationOperators.interpolateLinear(fanogram.getSubGrid(first_bucket), t);
						//float result_second = InterpolationOperators.interpolateLinear(fanogram.getSubGrid(second_bucket), t);
						//sinogram.putPixelValue(s, theta, (1 - (beta_degrees - beta_int)) * result_first + (beta_degrees - beta_int) * result_second);
						sinogram.putPixelValue(s, theta, result_first);
					} else {
						float result_first = fanogram.getSubGrid(first_bucket).getAtIndex(detector_f - 1);
						//float result_second = fanogram.getSubGrid(second_bucket).getAtIndex(detector_f - 1);
						//sinogram.putPixelValue(s, theta, (1 - (beta_degrees - beta_int)) * result_first + (beta_degrees - beta_int) * result_second);
						sinogram.putPixelValue(s, theta, result_first);
					}
				} else {
					if(gamma_index < ppd*opening_angle - 1) {
						float result_first = InterpolationOperators.interpolateLinear(fanogram.getSubGrid(first_bucket), gamma_index);
						//float result_second = InterpolationOperators.interpolateLinear(fanogram.getSubGrid(second_bucket), gamma_index);
						sinogram.putPixelValue(s, theta, result_first);
					} else {
						float result_first = fanogram.getSubGrid(first_bucket).getAtIndex(ppd*opening_angle - 1);
						//float result_second = fanogram.getSubGrid(second_bucket).getAtIndex(ppd*opening_angle - 1);
						sinogram.putPixelValue(s, theta, result_first);
					}
				}
			}
		}
		return sinogram;
	}
	
	
	public static void main(String[] args) {
		OurPhantom phantom = new OurPhantom();
		phantom.drawEllipse(phantom.width/2, phantom.width/2, (int) (phantom.width/2.5f), phantom.width/4, 0.5f);
		phantom.drawEllipse(phantom.width/2 + phantom.width/5, 5 * phantom.width/9, phantom.width/6, phantom.width/9, 0.3f);
		phantom.drawEllipse(phantom.width/2 - phantom.width/5, 5 * phantom.width/9, phantom.width/6, phantom.width/9, 0.3f);
		phantom.drawEllipse(phantom.width/2, phantom.width/2 - phantom.width/10, phantom.width/15, phantom.width/15, 0.7f);
		phantom.drawEllipse(phantom.width/2 - phantom.width/10, phantom.width/2 + phantom.width/10, phantom.width/25, phantom.width/30, 0.1f);
		
		ImageJ image = new ImageJ();
		
		phantom.show("Phantom");
		
		Grid2D sinogram = phantom.getSinogram();
//		sinogram.show("Sinogram");
		
		Grid2D fanogram = phantom.getFanogram(false, true);
//		fanogram.show("Fanogram - Full Scan");
		
		Grid2D fanogram_short = phantom.getFanogram(true, true);
//		fanogram_short.show("Fanogram - Short Scan");
		
		Grid2D sinogram_from_fanogram = phantom.rebin(fanogram, true);
//		sinogram_from_fanogram.show("Sinogram from Fanogram");
		
		Grid2D sinogram_from_fanogram_short = phantom.rebin(fanogram_short, true);
//		sinogram_from_fanogram_short.show("Sinogram from Short-Scan Fanogram");
		
		Grid2D filtered_sinogram_fourier = phantom.getFilteredSinogram_FourierDomain(sinogram);
//		filtered_sinogram_fourier.show();
		
		Grid2D filtered_sinogram_spatial = phantom.getFilteredSinogram_SpatialDomain(sinogram);
//		filtered_sinogram_spatial.show("Sinogram - filtered with Ram-Lak");
		
		Grid2D filtered_sinogram_from_fanogram_spatial = phantom.getFilteredSinogram_SpatialDomain(sinogram_from_fanogram);
//		filtered_sinogram_from_fanogram_spatial.show("Sinogram from Fanogram - filtered with Ram-Lak");
		
		Grid2D filtered_sinogram_from_fanogram_spatial_short = phantom.getFilteredSinogram_SpatialDomain(sinogram_from_fanogram_short);
//		filtered_sinogram_from_fanogram_spatial_short.show("Sinogram from Short-Scan Fanogram - filtered with Ram-Lak");
		
		Grid2D backprojected_image = phantom.getBackprojectionCL(sinogram);
		backprojected_image.show("CL backprojection of sinogram");
		
		Grid2D backprojected_image_fourier = phantom.getBackprojectionCL(filtered_sinogram_fourier);
		backprojected_image_fourier.show("CL backprojection of ramp-filtered sinogram");
		    
		Grid2D backprojected_image_spatial = phantom.getBackprojectionCL(filtered_sinogram_spatial);
		backprojected_image_spatial.show("CL backprojection of Ram-Lak filtered sinogram");
		
		Grid2D backprojected_image_from_fanogram_spatial = phantom.getBackprojectionCL(filtered_sinogram_from_fanogram_spatial);
		backprojected_image_from_fanogram_spatial.show("CL backprojection of sinogram from fanogram");
		
		Grid2D backprojected_image_from_fanogram_spatial_short = phantom.getBackprojectionCL(filtered_sinogram_from_fanogram_spatial_short);
		backprojected_image_from_fanogram_spatial_short.show("CL backprojection of short-scan sinogram from fanogram");
		
		
		
//		Grid2D old_phantom = new Grid2D(phantom);
//		float time_before_normal = System.nanoTime();
//		for(int t = 0; t < 100000; t++) {
//			for(int i = 0; i < phantom.getHeight(); i++) {
//				for(int j = 0; j < phantom.getWidth(); j++) {
//					phantom.addAtIndex(i, j, phantom.getAtIndex(i, j));
//				}
//			}
//		}
//		float time_after_normal = System.nanoTime();
//		System.out.println("Normal computation took " + (time_after_normal - time_before_normal)/1000000000 + " seconds.");
//		phantom.show();
//		
//		OpenCLGrid2D phantomCL = new OpenCLGrid2D(old_phantom);
//		float time_before_GPU = System.nanoTime();
//		for(int t = 0; t < 100000; t++) {
//			NumericPointwiseOperators.addBy(phantomCL, phantomCL);
//		}
//		float time_after_GPU = System.nanoTime();
//		System.out.println("GPU computation took " + (time_after_GPU - time_before_GPU)/1000000000 + " seconds.");
//		phantomCL.show();
		
		
		
//		OurPhantom phantom1 = new OurPhantom();
//		phantom1.drawEllipse(50, 70, 100, 60, 0.5f);
//		OurPhantom phantom2 = new OurPhantom();
//		phantom2.drawEllipse(100, 30, 70, 100, 0.3f);
//		phantom1.show();
//		phantom2.show();
//		
//		CLContext context = OpenCLUtil.getStaticContext();
//		CLDevice device = context.getMaxFlopsDevice();
//		CLProgram program = null;
//		try {
//			program = context.createProgram(OurPhantom.class.getResourceAsStream("Addition.cl")).build();
//		} catch(Exception e) {
//			System.err.println("Could not build program.");
//		}
//		
//		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16); 
//		int globalWorkSizeHeight = OpenCLUtil.roundUp(localWorkSize, phantom1.getHeight()); 
//		int globalWorkSizeWidth = OpenCLUtil.roundUp(localWorkSize, phantom1.getWidth());
//		
//		CLBuffer<FloatBuffer> phantom1Buffer = context.createFloatBuffer(phantom1.getWidth() * phantom1.getHeight(), Mem.READ_WRITE);
//		CLBuffer<FloatBuffer> phantom2Buffer = context.createFloatBuffer(phantom1.getWidth() * phantom1.getHeight(), Mem.READ_ONLY);
//		
//		for (int i = 0; i<phantom1.getNumberOfElements(); ++i){
//			phantom1Buffer.getBuffer().put(phantom1.getBuffer()[i]);
//		}
//		for (int i = 0; i<phantom2.getNumberOfElements(); ++i){
//			phantom2Buffer.getBuffer().put(phantom2.getBuffer()[i]);
//		}
//		phantom1Buffer.getBuffer().rewind();
//		phantom2Buffer.getBuffer().rewind();
//		
//		CLKernel kernelFunction = program.createCLKernel("addBy");
//		kernelFunction.putArg(phantom1Buffer).putArg(phantom2Buffer).putArg(phantom1.getWidth()).putArg(phantom1.getHeight());
//		
//		CLCommandQueue queue = device.createCommandQueue();
//		queue
//		.putWriteBuffer(phantom1Buffer, true)
//		.putWriteBuffer(phantom2Buffer, true)
//		.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSizeWidth, globalWorkSizeHeight,
//				localWorkSize, localWorkSize).finish()
//				.putReadBuffer(phantom1Buffer, true)
//				.finish();
//		
//		Grid2D img = new Grid2D(phantom1.getWidth(), phantom1.getHeight());
//		for (int i = 0; i < phantom1.getWidth() * phantom1.getHeight(); ++i) {
//				img.getBuffer()[i] = phantom1Buffer.getBuffer().get();
//		}
//		
//		img.show();
//		
//		queue.release();
//		phantom1Buffer.release();
//		phantom2Buffer.release();
//		kernelFunction.release();
//		program.release();
//		context.release();
	}
	
}
