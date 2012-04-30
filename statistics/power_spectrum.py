from numpy import *
from constants import *
from scipy import fftpack


def power_spectrum_nd(input_array):
	''' Calculate the power spectrum of input_array and return it as an n-dimensional array,
	where n is the number of dimensions in input_array'''


	print 'Calculating 2D power spectrum...'
	ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
	power_spectrum = abs(ft)**2
	print '...done'

	return power_spectrum

def cross_power_spectrum_nd(input_array1, input_array2):
	''' Calculate the cross power spectrum of input_array1 and input_array2 and return it as an n-dimensional array,
	where n is the number of dimensions in input_array'''

	assert(input_array1.shape == input_array2.shape)

	print 'Calculating 2D power spectrum...'
	ft1 = fftpack.fftshift(fftpack.fftn(input_array1.astype('float64')))
	ft2 = fftpack.fftshift(fftpack.fftn(input_array2.astype('float64')))
	power_spectrum = abs(ft1)*abs(ft2)
	print '...done'

	return power_spectrum


def radial_average(input_array, dim=2):
	''' Take an n-dimensional powerspectrum and return the radially averaged 
	version
	Return P(k), k (Mpc^-1)'''
	#TODO: use middle of k bins instead of edges, scale Pk correctly

	if dim == 2:
		y,x = indices(input_array.shape)
		center = array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
		r = hypot(x - center[0], y - center[1])
	elif dim == 3:
		y,x,z = indices(input_array.shape)
		center = array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0, (z.max()-z.min())/2.0])
		r = sqrt((x - center[0])**2 + (y - center[1])**2 + (z-center[2])**2)
	else:
		print 'Check your dimensions!'

	ind = argsort(r.flat) 			#indices of sorted array
	r_sorted = r.flat[ind] 			#sorted radii
	i_sorted = input_array.flat[ind] 	#sorted, flattened input array

	r_int = r_sorted.astype(int) 		#sorted radii as integers

	delta_r = r_int[1:]-r_int[:-1]		#differences in radius between successive points
	r_ind = where(delta_r)[0]		#Indices where delta_r != 0
	nr = r_ind[1:] - r_ind[:-1]		#Number of data points per step

	csum = cumsum(i_sorted, dtype='float64')	#Cumul. sum of sorted, flattened data
	tbin = csum[r_ind[1:]] - csum[r_ind[:-1]]#part of the csum falling at a given radius

	k_min=2.*pi/LB; #Mpc^-1
	k_max=k_min*(input_array.shape[0]/2.)*sqrt(3.);
	
	rvals = r_sorted[r_ind[1:]]
	k = (rvals-rvals.min())/rvals.max()*(k_max-k_min)+k_min

	##Test scale
	#boxvol = LB**len(input_array.shape)
	#pixelsize = boxvol/(product(input_array.shape))

	#return tbin/nr, r_sorted[r_ind[:-1]]
	return tbin.astype('float64')/nr.astype('float64'), k

def power_spectrum1d(input_array_nd):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k), k (in Mpc^-1)'''

	input_array = power_spectrum_nd(input_array_nd)	

	return radial_average(input_array, dim=len(input_array_nd.shape))

def cross_power_spectrum1d(input_array1_nd, input_array2_nd):
	''' Calculate the power spectrum of input_array_nd (2 or 3 dimensions)
	and return it as a one-dimensional array 
	Return P(k), k (in Mpc^-1)'''

	input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd)	

	return radial_average(input_array, dim=len(input_array1_nd.shape))


#TEST***************
if __name__ == '__main__':
	from density_file import *
	dfile = DensityFile('/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5/coarser_densities/halos_removed/7.059n_all.dat')
	#Pk,k = power_spectrum1d(dfile.raw_density)
	Pk,k = cross_power_spectrum1d(dfile.raw_density, dfile.raw_density)
	
	kanandata = loadtxt('/home/hjens/LyA/Power_spectrum/ps3dg.dat')

	import pylab as pl
	pl.loglog(k, Pk*k**3, 'r-')
	pl.loglog(kanandata[:,0], kanandata[:,0]**3*kanandata[:,2], 'b-')
	pl.show()
