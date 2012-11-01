import numpy as np
from .. import const
from .. import conv
from .. import utils
from scipy import fftpack
import pdb


def power_spectrum_nd(input_array, box_dims = None):
	''' Calculate the power spectrum of input_array and return it as an n-dimensional array,
	where n is the number of dimensions in input_array
	box_side is the size of the box in comoving Mpc. If this is set to None (default),
	the internal box size is used'''

	box_dims = get_dims(box_dims, input_array.shape)

	utils.print_msg( 'Calculating power spectrum...')
	ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
	power_spectrum = np.abs(ft)**2
	utils.print_msg( '...done')

	# scale
	boxvol = np.product(map(float,box_dims))
	pixelsize = boxvol/(np.product(input_array.shape))
	power_spectrum *= pixelsize**2/boxvol

	return power_spectrum

def cross_power_spectrum_nd(input_array1, input_array2, box_dims):
	''' Calculate the cross power spectrum of input_array1 and input_array2 and return it as an n-dimensional array,
	where n is the number of dimensions in input_array'''

	assert(input_array1.shape == input_array2.shape)

	box_dims = get_dims(box_dims, input_array1.shape)

	utils.print_msg( 'Calculating power spectrum...')
	ft1 = fftpack.fftshift(fftpack.fftn(input_array1.astype('float64')))
	ft2 = fftpack.fftshift(fftpack.fftn(input_array2.astype('float64')))
	power_spectrum = np.real(ft1)*np.real(ft2)+np.imag(ft1)*np.imag(ft2)
	utils.print_msg( '...done')

	# scale
	#boxvol = float(box_side)**len(input_array1.shape)
	boxvol = np.product(map(float,box_dims))
	pixelsize = boxvol/(np.product(map(float,input_array1.shape)))
	power_spectrum *= pixelsize**2/boxvol

	return power_spectrum


def radial_average(input_array, box_dims, kbins=10):
	'''
	Radially average the data in input_array
	box_side is the length of the box in Mpc
	kbins can be an integer specifying the number of bins,
	or a list of bins edges. if an integer is given, the bins
	are logarithmically spaced

	'''

	dim = len(input_array.shape)
	if dim == 2:
		x,y = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
		kx = 2.*np.pi * (x-center[0])/box_dims[0]
		ky = 2.*np.pi * (y-center[1])/box_dims[1]
		k = np.sqrt(kx**2 + ky**2)
	elif dim == 3:
		x,y,z = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0, (z.max()-z.min())/2.0])
		kx = 2.*np.pi * (x-center[0])/box_dims[0]
		ky = 2.*np.pi * (y-center[1])/box_dims[1]
		kz = 2.*np.pi * (z-center[2])/box_dims[2]
		k = np.sqrt(kx**2 + ky**2 + kz**2 ) 
	else:
		raise Exception('Check your dimensions!')

	if isinstance(kbins,int):
		kmin = 2.*np.pi/min(box_dims)
		kbins = 10**np.linspace(np.log10(kmin), np.log10(k.max()), bins+1)
	
	#Bin the data
	utils.print_msg('Binning data...')
	nbins = len(kbins)-1
	dk = (kbins[1:]-kbins[:-1])/2.
	outdata = np.zeros(nbins)
	for ki in range(nbins):
		kmin = kbins[ki]
		kmax = kbins[ki+1]
		idx = (k >= kmin) * (k < kmax)
		outdata[ki] = np.mean(input_array[idx])

	return outdata, kbins[:-1]+dk

	

def power_spectrum_1d(input_array_nd, kbins=100, box_dims=None):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	- input_array_nd is the data array
	- bins can be an array of k bin edges, a number of bins or None. If None is used,
	a faster binning algorithm is used, but the number and position of the bins are
	unspecified.
	Return P(k), k (in Mpc^-1)'''

	box_dims = get_dims(box_dims, input_array_nd.shape)

	input_array = power_spectrum_nd(input_array_nd, box_side=box_side)	

	return radial_average(input_array, kbins=kbins, box_side=box_side)

def cross_power_spectrum_1d(input_array1_nd, input_array2_nd, kbins=100, box_dims=None):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k) [Mpc^3], k [Mpc^-1]'''

	box_dims = get_dims(box_dims, input_array1_nd.shape)

	input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd, box_dims=box_dims)	

	return radial_average(input_array, kbins=kbins, box_dims = box_dims)

def power_spectrum_mu(input_array, los_axis = 0, mubins=20, kbins=10, box_dims = None, weights=None):
	'''
	Calculate the power spectrum and bin it in mu=cos(theta) and k
	input_array is the array to calculate the power spectrum from
	los_axis is the line of sight axis (default 0)
	mubins is the number of (linearly spaced) bins in mu or a list of bin edges
	kbins is the number of (log spaced) bins in k or a list of bin edges
	return Pk [Mpc^3] dim=(n_mubins,n_kbins), mu, k[Mpc^-1]
	'''

	#Calculate the power spectrum
	powerspectrum = power_spectrum_nd(input_array, box_dims=box_dims)	

	return mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights)

def cross_power_spectrum_mu(input_array1, input_array2, los_axis = 0, mubins=20, kbins=10, box_dims = None, weights=None):
	'''
	Calculate the cross power spectrum and bin it in mu=cos(theta) and k
	input_array1 and input_array2 are the arrays to calculate the power spectrum from
	los_axis is the line of sight axis (default 0)
	mubins is the number of (linearly spaced) bins in mu or a list of bin edges
	kbins is the number of (log spaced) bins in k or a list of bin edges
	return Pk [Mpc^3] dim=(n_mubins,n_kbins), mu, k[Mpc^-1]
	'''

	box_dims = get_dims(box_dims, input_array1.shape)

	#Calculate the power spectrum
	powerspectrum = cross_power_spectrum_nd(input_array1, input_array2, box_dims=box_dims)	

	return mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights)


def mu_binning(powerspectrum, los_axis = 0, mubins=20, kbins=10, box_dims = None, weights=None):
	'''
	Bin a power spectrum in mu and k. For internal use
	'''

	if weights != None:
		powerspectrum *= weights

	dim = len(powerspectrum.shape)
	assert(dim==3)

	x,y,z = np.indices(powerspectrum.shape)
	center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0, (z.max()-z.min())/2.0])
	kx = 2.*np.pi * (x-center[0])/box_dims[0]
	ky = 2.*np.pi * (y-center[1])/box_dims[1]
	kz = 2.*np.pi * (z-center[2])/box_dims[2]
	k = np.sqrt(kx**2 + ky**2 + kz**2 ) 

	#Line-of-sight distance from center 
	if los_axis == 0:
		los_dist = kx
	elif los_axis == 1:
		los_dist = ky
	elif los_axis == 2:
		los_dist = kz
	else:
		raise Exception('Your space is not %d-dimensional!' % los_axis)

	#mu=cos(theta) = k_par/k
	mu = los_dist/np.abs(k)
	mu[np.where(k < 0.001)] = np.nan

	#Calculate k values, and make bins
	if isinstance(kbins,int):
		kbins = 10**np.linspace(np.log10(k.min()),np.log10(k.max()),kbins+1)
	dk = (kbins[1:]-kbins[:-1])/2.
	n_kbins = len(kbins)-1

	#Exclude the k_x = 0, k_y = 0, k_z = 0 modes
	zero_ind = (x == k.shape[0]/2) + (y == k.shape[1]/2) + (z == k.shape[2]/2)
	k[zero_ind] = -1.

	#Make mu bins
	if isinstance(mubins,int):
		mubins = np.linspace(-1., 1., mubins+1)
	dmu = (mubins[1:]-mubins[:-1])/2.
	n_mubins = len(mubins)-1

	#Remove the zero component from the power spectrum. mu is undefined here
	powerspectrum[tuple(np.array(powerspectrum.shape)/2)] = 0.

	#Bin the data
	utils.print_msg('Binning data...')
	outdata = np.zeros((n_mubins,n_kbins))
	for ki in range(n_kbins):
		kmin = kbins[ki]
		kmax = kbins[ki+1]
		kidx = (k >= kmin) * (k < kmax)
		for i in range(n_mubins):
			mu_min = mubins[i]
			mu_max = mubins[i+1]
			idx = (mu >= mu_min) * (mu < mu_max) * kidx
			outdata[i,ki] = np.mean(powerspectrum[idx])

			if weights != None:
				outdata[i,ki] /= weights[idx].mean()

	return outdata, mubins[:-1]+dmu, kbins[:-1]+dk


def get_dims(box_dims, ashape):
	if box_dims == None:
		return [100]*len(ashape)
	if not hasattr(box_dims, '__iter__'):
		return [box_dims]*len(ashape)
	return box_dims
