import numpy as np
from .. import const
from .. import conv
from .. import utils
from scipy import fftpack


def power_spectrum_nd(input_array, box_side = None):
	''' Calculate the power spectrum of input_array and return it as an n-dimensional array,
	where n is the number of dimensions in input_array
	box_side is the size of the box in comoving Mpc. If this is set to None (default),
	the internal box size is used'''

	if box_side == None:
		box_side = conv.LB

	utils.print_msg( 'Calculating 2D power spectrum...')
	ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
	power_spectrum = np.abs(ft)**2
	utils.print_msg( '...done')

	# scale
	boxvol = box_side**len(input_array.shape)
	pixelsize = boxvol/(np.product(input_array.shape))
	print 'pixelsize:', pixelsize
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


def radial_average_fast(input_array, box_side, dim=2):
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

	if box_side == None:
		box_side = conv.LB

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

	k = 2.*np.pi/box_side*(rvals-dr/2.)

	return tbin.astype('float64')/nr.astype('float64'), k


def radial_average_flexible(input_array, box_side, dim=2, bins=10):
	''' This is a slightly slower version of the radial average routine above.
	It allows for more flexible binning however. '''

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

	#Calculate k values
	k = 2.*np.pi/box_side*r

	#If bins is None, do the fast binning
	if bins == None:
		return radial_average_fast(input_array, box_side)
	if isinstance(bins,int):
		kmin = 2.*np.pi/box_side
		bins = np.linspace(kmin, k.max(), bins+1)
	
	#Bin the data
	utils.print_msg('Binning data...')
	nbins = len(bins)-1
	dk = (bins[1:]-bins[:-1])/2.
	outdata = np.zeros(nbins)
	for ki in range(nbins):
		kmin = bins[ki]
		kmax = bins[ki+1]
		idx = (k >= kmin) * (k < kmax)
		outdata[ki] = np.mean(input_array[idx])

	return outdata, bins[:-1]+dk

	

def power_spectrum_1d(input_array_nd, bins=100, box_side=None):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	- input_array_nd is the data array
	- bins can be an array of k bin edges, a number of bins or None. If None is used,
	a faster binning algorithm is used, but the number and position of the bins are
	unspecified.
	Return P(k), k (in Mpc^-1)'''

	if box_side == None:
		box_side = conv.LB

	input_array = power_spectrum_nd(input_array_nd, box_side=box_side)	

	return radial_average_flexible(input_array, dim=len(input_array_nd.shape), bins=bins, box_side=box_side)

def cross_power_spectrum1d(input_array1_nd, input_array2_nd):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k) [Mpc^3], k [Mpc^-1]'''

	input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd)	

	return radial_average(input_array, dim=len(input_array1_nd.shape))

def power_spectrum_mu(input_array, los_axis = 0, n_mubins=20, n_kbins=10):
	'''
	Calculate the power spectrum and bin it in mu=cos(theta) and k
	input_array is the array to calculate the power spectrum from
	los_axis is the line of sight axis (default 0)
	mubins is the number of (linearly spaced) bins in mu
	kbins is the number of (linearly spaced) bins in k
	return Pk [Mpc^3] dim=(n_mubins,n_kbins), mu, k[Mpc^-1]
	TODO: allow custom box side, verify that this routine works at all
	'''

	print 'Warning! The mu binning is not tested. It probably doesn\'t work yet. You should not use it.'

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

	#Line-of-sight distance from center 
	if los_axis == 0:
		los_dist = x-center[0]
	elif los_axis == 1:
		los_dist = y-center[0]
	elif los_axis == 2:
		los_dist = z-center[0]
	else:
		raise Exception('Your space is not %d-dimensional!' % los_axis)

	#mu=cos(theta) = k_par/k
	mu = np.abs(los_dist)/np.abs(r)
	mu[np.where(r < 0.001)] = np.nan

	#Calculate k values
	k = 2.*np.pi/conv.LB*r
	kbins = np.linspace(k.min(),k.max(),n_kbins)
	dk = kbins[1]-kbins[0]
	kbins += dk/2.

	#Calculate the power spectrum
	powerspectrum = power_spectrum_nd(input_array)	

	#Remove the zero component from the power spectrum. mu is undefined here
	powerspectrum[tuple(np.array(input_array.shape)/2)] = 0.

	#Bin the data
	utils.print_msg('Binning data...')
	dmu = 1./float(n_mubins)
	mubins = np.linspace(dmu/2., 1.-dmu/2, n_mubins)
	outdata = np.zeros((n_mubins,n_kbins))
	for ki in range(n_kbins):
		kmin = kbins[ki]-dk/2.
		kmax = kbins[ki]+dk/2.
		for i in range(n_mubins):
			mu_min = mubins[i]-dmu/2.
			mu_max = mubins[i]+dmu/2.
			idx = (mu >= mu_min) * (mu < mu_max) * (k >= kmin) * (k < kmax)
			outdata[i,ki] = np.mean(powerspectrum[idx])

	return outdata, mubins, kbins

