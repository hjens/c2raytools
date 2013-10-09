import const
import conv
import numpy as np
import os
import glob
from temperature import calc_dt
from helper_functions import print_msg, get_dens_redshifts, get_mesh_size, \
	determine_redshift_from_filename, get_data_and_type
from xfrac_file import XfracFile
from density_file import DensityFile

def make_lightcone(filenames, z_low = None, z_high = None, file_redshifts = None, \
				cbin_bits = 32, cbin_order = 'c', los_axis = 0):
	'''
	Make a lightcone from xfrac, density or dT data. Replaces freq_box.
	
	Parameters:
		* filenames (string or array): The coeval cubes. 
			Can be either any of the following:
			
			 	- An array with the file names
			 	
			 	- A text file containing the file names
			 	
				- The directory containing the files (must only contain 
				one type of files)
		* z_low (float): the lowest redshift. If not given, the redshift of the 
			lowest-z coeval cube is used.
		* z_high (float): the highest redshift. If not given, the redshift of the 
			highest-z coeval cube is used.
		* file_redshifts (string or array): The redshifts of the coeval cubes.
			Can be any of the following types:
			
			- None: determine the redshifts from file names
			 
			- array: array containing the redshift of each coeval cube
			
			- filename: the name of a data file to read the redshifts from
			
		* cbin_bits (int): If the data files are in cbin format, you may specify 
			the number of bits.
		* cbin_order (char): If the data files are in cbin format, you may specify 
			the order of the data.
		* los_axis (int): the axis to use as line-of-sight for the coeval cubes
		
	Returns:
		(lightcone, z) tuple
		
		lightcone is the lightcone volume where the first two axes
		have the same size as the input cubes
		
		z is an array containing the redshifts along the line-of-sight
		
	.. note::
		If z_low is given, that redshift will be the lowest one included,
		even if there is no coeval box at exactly that redshift. This can 
		give results that are subtly different from results calculated with
		the old freq_box routine.
	'''
	
	filenames = _get_filenames(filenames)
	file_redshifts = _get_file_redshifts(file_redshifts, filenames)
	assert(len(file_redshifts) == len(filenames))
	mesh_size = get_mesh_size(filenames[0])
	
	if z_low == None:
		z_low = file_redshifts.min()
	if z_high == None:
		z_high = file_redshifts.max()
		
	output_z = redshifts_at_equal_comoving_distance(z_low, z_high, box_grid_n=mesh_size[0])
	if min(output_z) < min(file_redshifts) or max(output_z) > max(file_redshifts):
		print 'Warning! You have specified a redshift range of %.3f < z < %.3f' % (min(output_z), max(output_z))
		print 'but you only have files for the range %.3f < z < %.3f.' % (min(file_redshifts), max(file_redshifts))
		print 'The redshift range will be truncated.'
		output_z = output_z[output_z >= min(file_redshifts)]
		output_z = output_z[output_z <= max(file_redshifts)]
	if len(output_z) < 1:
		raise Exception('No valid redshifts in range!')

	lightcone = np.zeros((mesh_size[0], mesh_size[1], len(output_z)))
	
	comoving_pos_idx = 0
	z_bracket_low = None; z_bracket_high = None
	
	for z in output_z:
		z_bracket_low_new = file_redshifts[file_redshifts < z].max()
		z_bracket_high_new = file_redshifts[file_redshifts > z].min()
		
		if z_bracket_low_new != z_bracket_low:
			z_bracket_low = z_bracket_low_new
			file_idx = np.argmin(np.abs(file_redshifts - z_bracket_low))
			data_low, datatype = get_data_and_type(filenames[file_idx], cbin_bits, cbin_order)
			
		if z_bracket_high_new != z_bracket_high:
			z_bracket_high = z_bracket_high_new
			file_idx = np.argmin(np.abs(file_redshifts - z_bracket_high))
			data_high, datatype = get_data_and_type(filenames[file_idx], cbin_bits, cbin_order)
		
		data_interp = _get_interp_slice(data_high, data_low, z_bracket_high, \
									z_bracket_low, z, comoving_pos_idx, los_axis)
		lightcone[:,:,comoving_pos_idx] = data_interp
		
		comoving_pos_idx += 1
		
	return lightcone, output_z


def redshifts_at_equal_comoving_distance(z_low, z_high, box_grid_n=256, \
			box_length_mpc = None):
	''' 
	Make a frequency axis vector with equal spacing in co-moving LOS coordinates. 
	The comoving distance between each frequency will be the same as the cell
	size of the box.
	
	Parameters:
		* z_low (float): The lower redshift
		* z_high (float): The upper redhisft 
		* box_grid_n = 256 (int): the number of slices in an input box
		* box_length_mpc (float): the size of the box in cMpc. If None,
		set to conv.LB
			 
	Returns:
		numpy array containing the redshifts
		
	'''
	if box_length_mpc == None:
		box_length_mpc = conv.LB
	assert(z_high > z_low)

	z = z_low
	z_array = []
	nu_array = []

	while z < z_high:
		nu = const.nu0/(1.0+z)

		z_array.append(z)
		nu_array.append(nu)

		dnu = const.nu0*const.Hz(z)*box_length_mpc/(1.0 + z)**2/const.c/float(box_grid_n)

		z = const.nu0/(nu - dnu) - 1.0

	return np.array(z_array)


def _get_interp_slice(data_high, data_low, z_bracket_high, z_bracket_low, z, comoving_pos_idx, los_axis):
	slice_ind = comoving_pos_idx % data_low.shape[0]
	#slice_low = data_low[slice_ind,:,:]
	#slice_high = data_high[slice_ind,:,:]
	slice_low = _get_slice(data_low, slice_ind, los_axis)
	slice_high = _get_slice(data_high, slice_ind, los_axis)
	slice_interp = ((z-z_bracket_low)*slice_high + (z_bracket_high - z)*slice_low)/(z_bracket_high-z_bracket_low)
	
	return slice_interp


def _get_slice(data, idx, los_axis):
	assert(len(data.shape) == 3)
	assert(los_axis >= 0 and los_axis < len(data.shape))
	if los_axis == 0:
		return data[idx,:,:]
	elif los_axis == 1:
		return data[:,idx,:]
	
	return data[:,:,idx]
	
	
def _get_filenames(filenames_in):
	'''
	If filenames_in is a list of files, return as it is
	If it is a directory, make sure it only contains data files,
	then return the list of files in the directory
	If it is a text file, read the list of files from the file
	'''
	
	if hasattr(filenames_in, '__iter__'):
		filenames_out = filenames_in
	elif os.path.isdir(filenames_in):
		files_in_dir = glob.glob(filenames_in + '/*')
		extensions = [os.path.splitext(f)[-1] for f in files_in_dir]
		if not _all_same(extensions):
			raise Exception('The directory may only contain one file type.')
		filenames_out = files_in_dir
	else:
		f = open(filenames_in)
		names = [l.strip() for l in f.readlines()]
		f.close()
		filenames_out = names
		
	return np.array(filenames_out)

	
def _get_file_redshifts(redshifts_in, filenames):
	'''
	If redshifts_in is None, try to determine from file names
	If it's a directory, read the redshifts
	Else, return as is
	'''
	
	if hasattr(redshifts_in, '__iter__'):
		redshifts_out = redshifts_in
	elif redshifts_in == None:
		redshifts_out = [determine_redshift_from_filename(f) for f in filenames]
		redshifts_out = np.array(redshifts_out)
	elif os.path.exists(redshifts_in):
		redshifts_out = np.loadtxt(redshifts_in)
	else:
		raise Exception('Invalid data for file redshifts.')
	
	return redshifts_out


def _all_same(items):
	return all(x == items[0] for x in items)

	
