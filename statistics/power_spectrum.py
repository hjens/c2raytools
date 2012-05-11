import numpy as np
from .. import const
from .. import conv
from .. import utils
from scipy import fftpack


def power_spectrum_nd(input_array):
	''' Calculate the power spectrum of input_array and return it as an n-dimensional array,
	where n is the number of dimensions in input_array'''


	utils.print_msg( 'Calculating 2D power spectrum...')
	ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
	power_spectrum = np.abs(ft)**2
	utils.print_msg( '...done')

	# scale
	boxvol = conv.LB**len(input_array.shape)
	pixelsize = boxvol/(np.product(input_array.shape))
	power_spectrum *= pixelsize**2/boxvol

	return power_spectrum

def cross_power_spectrum_nd(input_array1, input_array2):
	''' Calculate the cross power spectrum of input_array1 and input_array2 and return it as an n-dimensional array,
	where n is the number of dimensions in input_array'''

	assert(input_array1.shape == input_array2.shape)

	utils.print_msg( 'Calculating 2D power spectrum...')
	ft1 = fftpack.fftshift(fftpack.fftn(input_array1.astype('float64')))
	ft2 = fftpack.fftshift(fftpack.fftn(input_array2.astype('float64')))
	power_spectrum = np.abs(ft1)*np.abs(ft2)
	utils.print_msg( '...done')

	return power_spectrum




def radial_average(input_array, dim=2, nbins=0):
	''' Take an n-dimensional powerspectrum and return the radially averaged 
	version. For internal use mostly.
	Return P(k), k (Mpc^-1)'''

	if dim == 2:
		y,x = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
		r = np.hypot(x - center[0], y - center[1])
	elif dim == 3:
		y,x,z = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0, (z.max()-z.min())/2.0])
		r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z-center[2])**2)
	else:
		raise Exception( 'Check your dimensions!')

	ind = np.argsort(r.flat) 			#indices of sorted array
	r_sorted = r.flat[ind] 			#sorted radii
	i_sorted = input_array.flat[ind] 	#sorted, flattened input array

	r_int = r_sorted.astype(int) 		#sorted radii as integers

	delta_r = r_int[1:]-r_int[:-1]		#differences in radius between successive points
	r_ind = np.where(delta_r)[0]		#Indices where delta_r != 0
	nr = r_ind[1:] - r_ind[:-1]		#Number of data points per step

	csum = np.cumsum(i_sorted, dtype='float64')	#Cumul. sum of sorted, flattened data
	tbin = csum[r_ind[1:]] - csum[r_ind[:-1]]#part of the csum falling at a given radius

	rvals = r_sorted[r_ind[1:]]
	dr = rvals[1]-rvals[0]

	k = 2.*np.pi/conv.LB*(rvals-dr/2.)

	return tbin.astype('float64')/nr.astype('float64'), k
	

def power_spectrum_1d(input_array_nd, nbins=100):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k), k (in Mpc^-1)'''


	input_array = power_spectrum_nd(input_array_nd)	

	return radial_average(input_array, dim=len(input_array_nd.shape), nbins=nbins)

def cross_power_spectrum1d(input_array1_nd, input_array2_nd):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k), k (in Mpc^-1)'''

	input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd)	

	return radial_average(input_array, dim=len(input_array1_nd.shape))

def power_spectrum_mu(input_array, nbins=20):
	'''
	Calculate the power spectrum and bin it in mu=cos(theta)
	'''

	dim = len(input_array.shape)
	if dim == 2:
		y,x = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
		r = np.hypot(x - center[0], y - center[1])
	elif dim == 3:
		y,x,z = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0, (z.max()-z.min())/2.0])
		r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z-center[2])**2)
	else:
		raise Exception('Check your dimensions!')

	#Line-of-sight distance from center (assume los_axis = 0 for now)
	los_dist = x-center[0]

	#mu=cos(theta) = k_par/k
	mu = np.abs(los_dist)/np.abs(r)
	mu[np.where(r < 0.001)] = np.nan

	#Calculate the power spectrum
	powerspectrum = power_spectrum_nd(input_array)	
	powerspectrum[tuple(np.array(input_array.shape)/2)] = 0.

	#Bin the data
	dmu = 1./float(nbins)
	mubins = np.linspace(dmu/2., 1.-dmu/2, nbins)
	mudata = np.zeros(nbins)
	working_mask = np.ones(input_array.shape, bool)
	for i in range(nbins):
		mu_min = mubins[i]-dmu/2.
		mu_max = mubins[i]+dmu/2.
		idx = (mu >= mu_min) * (mu < mu_max) * working_mask
		print mu_min, mu_max, len(idx), np.mean(powerspectrum[idx])
		mudata[i] = np.mean(powerspectrum[idx])

	return mudata, mubins

