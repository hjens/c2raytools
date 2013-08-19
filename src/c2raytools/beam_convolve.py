import numpy as np
import const
from cosmology import zang
from scipy import signal
from helper_functions import print_msg
from misc import gauss_kern

def beam_convolve(input_array, z, fov_mpc, beam_w = None, max_baseline = None, \
				beamshape='gaussian'):
	''' 
	Convolve input_array with a beam of the specified form.
	The beam can be specified either by a width in arcminutes,
	or as a maximum baseline. You must specify exctly one of these
	parameters.
	
	Parameters:
		* input_array (numpy array): the array to be convolved
		* z (float): the redshift of the map
		* fov_mpc (float): the field of view in cMpc
		* beam_w (float) = None: the width of the beam in arcminutes
		* max_baseline (float): the maximum baseline in meters 
			(can be specified instead of beam_w)
		* beamshape (string): The shape of the beam 
			(only 'gaussian' supported at this time)
	
	Returns:
		The convolved array (a numpy array with the same dimensions
		as input_array).
	'''

	if (not beam_w) and (not max_baseline):
		raise Exception('Please specify either a beam width or a maximum baseline')
	elif not beam_w: #Calculate beam width from max baseline
		fr = const.nu0 / (1.0+z) 
		lw = const.c/fr/1.e6*1.e3 # wavelength in m
		beam_w = lw/max_baseline/np.pi*180.*60.

	angle = zang(fov_mpc*1000./(1.0 + z), z)/60.
	mx = input_array.shape[0]

	print_msg('Field of view is %.2f arcminutes' % (angle) )
	print_msg('Convolving with %s beam of size %.2f arcminutes...' % \
				(beamshape, beam_w) )

	#Convolve with beam
	if beamshape == 'gaussian':
		sigma0 = (beam_w)/angle/(2.0 * np.sqrt(2.0*np.log(2.)))*mx
		kernel = gauss_kern(sigma=sigma0, size=mx)
	else:
		raise Exception('Unknown beamshape: %g' % beamshape)

	out =  signal.fftconvolve(input_array, kernel)

	#fftconvolve makes the output twice the size, so return only the central part	
	ox = out.shape[0]
	return out[ox*0.25:ox*0.75, ox*0.25:ox*0.75]



