import numpy as np
import const
import scipy.ndimage as ndimage
import scipy.interpolate
from scipy import signal
from scipy.fftpack import fft, ifft, fftn, ifftn
from numpy.fft import rfftn, irfftn
from math import ceil, floor
from numpy import array, asarray, rank, roll
from helper_functions import fftconvolve

def gauss_kernel(size, sigma=1.0, fwhm=None):
	''' 
	Generate a normalized gaussian kernel, defined as
	exp(-(x^2 + y^2)/(2sigma^2)).
	
	
	Parameters:
		* size (int): Width of output array in pixels.
		* sigma = 1.0 (float): The sigma parameter for the Gaussian.
		* fwhm = None (float or None): The full width at half maximum.
				If this parameter is given, it overrides sigma.
		
	Returns:
		numpy array with the Gaussian. The dimensions will be
		size x size or size x sizey depending on whether
		sizey is set. The Gaussian is normalized so that its
		integral is 1.	
	'''
	
	if fwhm != None:
		sigma = fwhm/(2.*np.sqrt(2.*np.log(2)))

	if size % 2 == 0:
		size = int(size/2)
		x,y = np.mgrid[-size:size, -size:size]
	else:
		size = int(size/2)
		x,y = np.mgrid[-size:size+1, -size:size+1]
	
	g = np.exp(-(x**2 + y**2)/(2.*sigma**2))

	return g/g.sum()


def tophat_kernel(size, tophat_width):
	'''
	Generate a square tophat kernel
	
	Parameters:
		* size (int): the size of the array
		* tophat_width (int): the size of the tophat kernel
		
	Returns:
		The kernel as a (size,size) array
	'''
	kernel = np.zeros((size,size))
	center = kernel.shape[0]/2
	idx_low = center-np.floor(tophat_width/2.)
	idx_high = center+np.ceil(tophat_width/2.)
	kernel[idx_low:idx_high, idx_low:idx_high] = 1.
	kernel /= np.sum(kernel)
	return kernel


def tophat_kernel_3d(size):
	'''
	Generate a 3-dimensional tophat kernel with
	the specified size
	
	Parameters:
		* size (integer or list-like): the size of
			the tophat kernel along each dimension. If
			size is an integer, the kernel will be cubic.
	Returns:
		The normalized kernel
	'''
	if hasattr(size, '__iter__'):
		kernel = np.ones(size)
	else: #Integer
		kernel = np.ones((size, size, size))
	kernel /= np.sum(kernel)
	return kernel


def lanczos_kernel(size, kernel_width):
	'''
	Generate a 2D Lanczos kernel.
	
	Parameters:
		* size (int): the size of the array
		* kernel_width (int): the width of the kernel
		
	Returns:
		The kernel as a (size,size) array

	'''
	#x,y = np.mgrid[-size*0.5:size*0.5, -size*0.5:size*0.5]
	xi = np.linspace(-size*0.5, size*0.5, size)
	yi = np.linspace(-size*0.5, size*0.5, size)
	x, y = np.meshgrid(xi, yi)
	a = kernel_width
	kernel = np.sinc(x)*np.sinc(x/a)*np.sinc(y)*np.sinc(y/a)
	kernel[np.abs(x) > a] = 0.
	kernel[np.abs(y) > a] = 0.
	kernel /= kernel.sum()
	
	return kernel


def smooth_gauss(input_array, sigma):
	''' 
	Smooth the input array with a Gaussian kernel.
	
	Parameters:
		* input_array (numpy array): the array to smooth
		* sigma (float): the width of the kernel

	Returns:
		The smoothed array. A numpy array with the same
		dimensions as the input.
	'''
	kernel = gauss_kernel(input_array.shape[0], sigma=sigma)
	return smooth_with_kernel(input_array, kernel)


def smooth_tophat(input_array, tophat_width):
	''' 
	Smooth the input array with a square tophat kernel.
	
	Parameters:
		* input_array (numpy array): the array to smooth
		* tophat_width (int): the width of the kernel in cells

	Returns:
		The smoothed array. A numpy array with the same
		dimensions as the input.
	'''
	#For some reason fftconvolve works produces edge effects with
	#an even number of cells, so we pad the array with an extra pixel
	#if this is the case
	if input_array.shape[0] % 2 == 0:
		from angular_coordinates import _get_padded_slice
		padded = _get_padded_slice(input_array, input_array.shape[0]+1)
		out = smooth_tophat(padded, tophat_width)
		return out[:-1,:-1]
	
	kernel = tophat_kernel(input_array.shape[0], tophat_width)
	return smooth_with_kernel(input_array, kernel)


def smooth_lanczos(input_array, kernel_width):
	''' 
	Smooth the input array with a Lanczos kernel.
	
	Parameters:
		* input_array (numpy array): the array to smooth
		* kernel_width (int): the width of the kernel in cells

	Returns:
		The smoothed array. A numpy array with the same
		dimensions as the input.
	'''

	kernel = lanczos_kernel(input_array.shape[0], kernel_width)
	return smooth_with_kernel(input_array, kernel)


def smooth_with_kernel(input_array, kernel):
	''' 
	Smooth the input array with an arbitrary kernel.
	
	Parameters:
		* input_array (numpy array): the array to smooth
		* kernel (numpy array): the smoothing kernel. Must
			be the same size as the input array

	Returns:
		The smoothed array. A numpy array with the same
		dimensions as the input.
	'''
	assert len(input_array.shape) == len(kernel.shape)
	
	out = fftconvolve(input_array, kernel)
	
	return out


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
			is 0 (linear interpolation). Setting order=1 gives the same
			behaviour as IDL's interpolate function with default parameters.

	Returns:
		Interpolated array with shape (nx, ny, nz), where nx, ny and nz
		are the lengths of the arrays x, y and z respectively.
	'''
	
	
	inds = np.zeros((3, len(x), len(y), len(z)))
	inds[0,:,:] = x[:,np.newaxis,np.newaxis]
	inds[1,:,:] = y[np.newaxis,:,np.newaxis]
	inds[2,:,:] = z[np.newaxis,np.newaxis,:]
	new_array = ndimage.map_coordinates(input_array, inds, mode='wrap', \
									order=order)
	
	return new_array


def interpolate2d(input_array, x, y, order=0):
	'''
	Same as interpolate2d but for 2D data
	
	Parameters:
		* input_array (numpy array): the array to interpolate
		* x (numpy array): the output coordinates along the x axis
			expressed as (fractional) indices 
		* y (numpy array): the output coordinates along the y axis
			expressed as (fractional) indices 
		* order (int): the order of the spline interpolation. Default
			is 0 (linear interpolation). Setting order=1 gives the same
			results as IDL's interpolate function

	Returns:
		Interpolated array with shape (nx, ny), where nx and ny
		are the lengths of the arrays x and y respectively.
	'''

	inds = np.zeros((2, len(x), len(y)))
	inds[0,:] = x[:,np.newaxis]
	inds[1,:] = y[np.newaxis,:]
	new_array = ndimage.map_coordinates(input_array, inds, mode='wrap', \
									order=order, prefilter=True)
	
	return new_array




