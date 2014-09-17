import numpy as np
import const
import scipy.ndimage as ndimage


def gauss_kern(size, sizey = None, sigma=1.0):
	''' 
	Generate a normalized gaussian kernel. 
	
	Parameters:
		* size (int): Width of output array in pixels.
		* sizey = None (int): Width along the x axis. If this
			is set to None, sizey = size
		* sigma = 1.0 (float): The sigma parameter for the Gaussian.
		
	Returns:
		numpy array with the Gaussian. The dimensions will be
		size x size or size x sizey depending on whether
		sizey is set. The Gaussian is normalized so that its
		integral is 1.	
	'''

	size = int(size/2)
	if not sizey:
		sizey = size
	else:
		sizey = int(sizey/2)

	x,y = np.mgrid[-size:size, -sizey:sizey]
	g = np.exp(-(x**2 + y**2)/sigma**2)

	return g/g.sum()


def smooth(input_array, sigma):
	''' 
	Smooth the input array with a Gaussian kernel.
	
	Parameters:
		* input_array (numpy array): the array to smooth
		* sigma (float): the width of the kernel

	Returns:
		The smoothed array. A numpy array with the same
		dimensions as the input.
	'''
	from scipy import signal

	assert(input_array.shape[0] == input_array.shape[1])


	kernel = gauss_kern(input_array.shape[0], sigma=sigma)
	
	out =  signal.fftconvolve(input_array, kernel)

	#fftconvolve makes the array larger - return only
	#the central part
	ox = out.shape[0]

	return out[ox*0.25:ox*0.75, ox*0.25:ox*0.75]


def get_beam_w(baseline, z):
	'''
	Calculate the width of the beam for an
	interferometer with a given maximum baseline.
	It is assumed that observations are done at
	lambda = 21*(1+z) cm
	
	Parameters:
		* baseline (float): the maximum baseline in meters
		* z (float): the redshift
		
	Returns:
		The beam width in arcminutes
	'''
	
	fr = const.nu0 / (1.0+z) #21 cm frequency at z
	lw = const.c/fr/1.e6*1.e3 # wavelength in m
	beam_w = lw/baseline/np.pi*180.*60.
	return beam_w


def interpolate3d(input_array, x, y, z, order=0):
	'''
	This function is a recreation of IDL's interpolate
	routine. It takes an input array, and interpolates it
	to a new size, which can be irregularly spaced.
	
	Parameters:
		* input_array (numpy array): the array to interpolate
		* x (numpy array): the output coordinates along the x axis
			expressed as (fractional) indices 
		* y (numpy array): the output coordinates along the y axis
			expressed as (fractional) indices 
		* z (numpy array): the output coordinates along the z axis
			expressed as (fractional) indices
		* order (int): the order of the spline interpolation. Default
			is 0 (linear interpolation). 

	Returns:
		Interpolated array with shape (nx, ny, nz), where nx, ny and nz
		are the lengths of the arrays x, y and z respectively.
	'''
	
	
	inds = np.zeros((3, len(x), len(y), len(z)))
	inds[0,:,:] = x[:,np.newaxis,np.newaxis]
	inds[1,:,:] = y[np.newaxis,:,np.newaxis]
	inds[2,:,:] = z[np.newaxis,np.newaxis,:]
	new_array = ndimage.map_coordinates(input_array, inds, mode='wrap', order=0)
	
	return new_array
