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

	utils.print_msg( 'Calculating power spectrum...')
	ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
	power_spectrum = np.abs(ft)**2
	utils.print_msg( '...done')

	# scale
	boxvol = box_side**len(input_array.shape)
	pixelsize = boxvol/(np.product(input_array.shape))
	power_spectrum *= pixelsize**2/boxvol

	return power_spectrum

def cross_power_spectrum_nd(input_array1, input_array2, box_side = None):
	''' Calculate the cross power spectrum of input_array1 and input_array2 and return it as an n-dimensional array,
	where n is the number of dimensions in input_array'''

	if box_side == None:
		box_side = conv.LB

	assert(input_array1.shape == input_array2.shape)

	utils.print_msg( 'Calculating power spectrum...')
	ft1 = fftpack.fftshift(fftpack.fftn(input_array1.astype('float64')))
	ft2 = fftpack.fftshift(fftpack.fftn(input_array2.astype('float64')))
	power_spectrum = np.real(ft1)*np.real(ft2)+np.imag(ft1)*np.imag(ft2)
	utils.print_msg( '...done')

	# scale
	boxvol = box_side**len(input_array1.shape)
	pixelsize = boxvol/(np.product(input_array1.shape))
	power_spectrum *= pixelsize**2/boxvol

	return power_spectrum


def radial_average(input_array, box_side = None, bins=10):
	'''
	Radially average the data in input_array
	box_side is the length of the box in Mpc
	bins can be an integer specifying the number of bins,
	or a list of bins edges. if an integer is given, the bins
	are logarithmically spaced
	'''

	if box_side == None:
		box_side = conv.LB

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

	if isinstance(bins,int):
		kmin = 2.*np.pi/box_side
		#bins = np.linspace(kmin, k.max(), bins+1)
		bins = 10**np.linspace(np.log10(kmin), np.log10(k.max()), bins+1)
	
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

	return radial_average(input_array, bins=bins, box_side=box_side)

def cross_power_spectrum_1d(input_array1_nd, input_array2_nd, bins=100, box_side=None):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k) [Mpc^3], k [Mpc^-1]'''

	if box_side == None:
		box_side = conv.LB

	input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd, box_side=box_side)	

	return radial_average(input_array, bins=bins, box_side = box_side)

def power_spectrum_mu(input_array, los_axis = 0, mubins=20, kbins=10, box_side = None, weights=None):
	'''
	Calculate the power spectrum and bin it in mu=cos(theta) and k
	input_array is the array to calculate the power spectrum from
	los_axis is the line of sight axis (default 0)
	mubins is the number of (linearly spaced) bins in mu or a list of bin edges
	kbins is the number of (log spaced) bins in k or a list of bin edges
	return Pk [Mpc^3] dim=(n_mubins,n_kbins), mu, k[Mpc^-1]
	TODO: allow custom box side
	'''

	if box_side == None:
		box_side = conv.LB

	dim = len(input_array.shape)
	if dim == 2:
		x,y = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
		r = np.hypot(x - center[0], y - center[1])
	elif dim == 3:
		x,y,z = np.indices(input_array.shape)
		center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0, (z.max()-z.min())/2.0])
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
	mu = los_dist/np.abs(r)
	mu[np.where(r < 0.001)] = np.nan

	#Calculate k values, and make bins
	k = 2.*np.pi/box_side*r
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

	#Calculate the power spectrum
	powerspectrum = power_spectrum_nd(input_array, box_side=box_side)	

	if weights != None:
		powerspectrum *= weights

	#Remove the zero component from the power spectrum. mu is undefined here
	powerspectrum[tuple(np.array(input_array.shape)/2)] = 0.

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

