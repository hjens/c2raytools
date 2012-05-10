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


#def radial_average(input_array, dim=2, nbins=100):
#	''' Take an n-dimensional powerspectrum and return the radially averaged 
#	version
#	Return P(k), k (Mpc^-1)'''
#	#TODO: use middle of k bins instead of edges, scale Pk correctly
#
#	print 'WARNING: the power spectrum scaling is probably wrong...'
#
#	utils.print_msg('Radial average...')
#
#	if dim == 2:
#		y,x = np.indices(input_array.shape)
#		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
#		r = np.hypot(x - center[0], y - center[1])
#	elif dim == 3:
#		y,x,z = np.indices(input_array.shape)
#		center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0, (z.max()-z.min())/2.0])
#		r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z-center[2])**2)
#	else:
#		raise Exception('Check your dimensions!')
#
#	dr = r.max()/float(nbins)
#	radial = np.arange(r.max()/dr)*dr + dr/2.
#	nrad = len(radial)
#
#	working_mask = np.ones(input_array.shape, bool)
#	radialdata = np.zeros(nrad)
#	
#
#	for irad in range(nrad): 
#		percent_done = int(float(irad)/float(nrad)*100)
#		if percent_done%10 == 0:
#			utils.print_msg('%d %%' % percent_done)
#
#		minrad = irad*dr
#		maxrad = minrad + dr
#		thisindex = (r>=minrad) * (r<maxrad) * working_mask
#		if not thisindex.ravel().any():
#			radialdata[irad] = np.nan
#		else:
#			radialdata[irad] = input_array[thisindex].mean()
#
#	k = 2.*np.pi/conv.LB*radial
#
#	return radialdata, k


def radial_average(input_array, dim=2, nbins=0):
	''' Take an n-dimensional powerspectrum and return the radially averaged 
	version. For internal use mostly.
	Return P(k), k (Mpc^-1)'''
	#TODO: use middle of k bins instead of edges, scale Pk correctly

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

	#k_min=2.*np.pi/conv.LB; #Mpc^-1
	#k_max=k_min*(input_array.shape[0]/2.)*np.sqrt(3.);
	
	rvals = r_sorted[r_ind[1:]]
	#k = (rvals-rvals.min())/rvals.max()*(k_max-k_min)+k_min

	k = 2.*np.pi/conv.LB*rvals

	return tbin.astype('float64')/nr.astype('float64'), k
	

def power_spectrum_1d(input_array_nd, bins=100):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k), k (in Mpc^-1)'''


	input_array = power_spectrum_nd(input_array_nd)	

	# scale
	boxvol = conv.LB**len(input_array.shape)
	pixelsize = boxvol/(np.product(input_array.shape))
	input_array *= pixelsize/boxvol

	return radial_average(input_array, dim=len(input_array_nd.shape), nbins=bins)

def cross_power_spectrum1d(input_array1_nd, input_array2_nd):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k), k (in Mpc^-1)'''

	input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd)	

	return radial_average(input_array, dim=len(input_array1_nd.shape))

def power_spectrum_mu(input_array, mubins):
	pass



