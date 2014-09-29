import numpy as np
import const
import scipy.ndimage as ndimage
import scipy.interpolate

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
	new_array = ndimage.map_coordinates(input_array, inds, mode='wrap', order=order)
	
	return new_array


def resample_array(a, newdims, method='linear', centre=False, minusone=False):
    '''
    This method was stolen from the internet. Arbitrary resampling of 
    source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine.

	Parameters:
    	* method (string):
    		neighbour - closest value from original data
    		nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
   			spline - uses ndimage.map_coordinates

    	* centre (bool):
    		True - interpolation points are at the centres of the bins
    		False - points are at the front edge of the bin

    	*minusone (bool):
    		For example- inarray.shape = (i,j) & new dimensions = (x,y)
    		False - inarray is resampled by factors of (i/x) * (j/y)
    		True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    		his prevents extrapolation one element beyond bounds of input array.
    
    Returns:
    	The new array, with dimensions newdims
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported."
        return None




