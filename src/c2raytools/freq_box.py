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


def redshifts_at_equal_comoving_distance(z_low, z_high, box_grid_n=256, \
			box_length_mpc = conv.LB):
	''' 
	Make a frequency axis vector with equal spacing in co-moving LOS coordinates. 
	The comoving distance between each frequency will be the same as the cell
	size of the box.
	
	Parameters:
		* z_low (float): The lower redshift
		* z_high (float): The upper redhisft 
		* box_grid_n = 256 (int): the number of slices in an input box
		* box_length_mpc = conv.LB (float): the size of the box in cMpc
			 
	Returns:
		numpy array containing the redshifts
		
	'''
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


def freq_box(xfrac_dir, dens_dir, z_low, z_high):
	''' 
	Make frequency (lightcone) boxes of density, ionized fractions, 
	and brightness temperature. The function reads xfrac and density
	files from the specified directories and combines them into a 
	lighcone box going from z_low to z_high.
	
	This routine is more or less a direct translation of Garrelt's 
	IDL routine.
	
	Parameters: 
		* xfrac_dir (string): directory containing xfrac files
		* dens_dir (string): directory containing density files
		* z_low (float): lowest redshift to include
		* z_high (float): highest redshift to include.

	Returns: 
		Tuple with (density box, xfrac box, dt box, redshifts), where
		density box, xfrac box and dt box are numpy arrays containing
		the lightcone quantities. redshifts is an array containing the 
		redshift for each slice.
		
	.. note::
		Since this function relies on filenames to get redshifts,
		all the data files must follow the common naming convenstions.
		Ionization files must be named xfrac3d_z.bin and densityfiles
		zn_all.dat
		
	Example:
		Make a lightcone cube ranging from z = 7 to z = 8:
	
		>>> xfrac_dir = '/path/to/data/xfracs/'
		>>> dens_dir = '/path/to/data/density/'
		>>> xcube, dcube, dtcube, z = c2t.freq_box(xfrac_dir, density_dir, z_low=7.0, z_high=8.)
		
	'''
	

	#Get the list of redshifts where we have simulation output files
	dens_redshifts = get_dens_redshifts(dens_dir, z_low )
	mesh_size = get_mesh_size(os.path.join(dens_dir, '%.3fn_all.dat' % dens_redshifts[0]))

	#Get the list of redhifts and frequencies that we want for the observational box
	output_z = redshifts_at_equal_comoving_distance(z_low, z_high, box_grid_n=mesh_size[0])
	output_z = output_z[output_z > dens_redshifts[0]]
	output_z = output_z[output_z < dens_redshifts[-1]]
	if len(output_z) < 1:
		raise Exception('No valid redshifts in range!')

	#Keep track of output simulation files to use
	xfrac_file_low = XfracFile(); xfrac_file_high = XfracFile()
	dens_file_low = DensityFile(); dens_file_high = DensityFile()
	z_bracket_low = None; z_bracket_high = None

	#The current position in comoving coordinates
	comoving_pos_idx = 0

	#Build the cube
	xfrac_lightcone = np.zeros((mesh_size[0], mesh_size[1], len(output_z)))
	dens_lightcone = np.zeros_like(xfrac_lightcone)
	dt_lightcone = np.zeros_like(xfrac_lightcone)
	
	for z in output_z:
		#Find the output files that bracket the redshift
		z_bracket_low_new = dens_redshifts[dens_redshifts <= z][0]
		z_bracket_high_new = dens_redshifts[dens_redshifts >= z][0]

		if z_bracket_low_new != z_bracket_low:
			z_bracket_low = z_bracket_low_new
			xfrac_file_low = XfracFile(os.path.join(xfrac_dir, 'xfrac3d_%.3f.bin' % z_bracket_low))
			dens_file_low = DensityFile(os.path.join(dens_dir, '%.3fn_all.dat' % z_bracket_low))
			dt_cube_low = calc_dt(xfrac_file_low, dens_file_low)

		if z_bracket_high_new != z_bracket_high:
			z_bracket_high = z_bracket_high_new
			xfrac_file_high = XfracFile(os.path.join(xfrac_dir, 'xfrac3d_%.3f.bin' % z_bracket_high))
			dens_file_high = DensityFile(os.path.join(dens_dir, '%.3fn_all.dat' % z_bracket_high))
			dt_cube_high = calc_dt(xfrac_file_high, dens_file_high)
			
		slice_ind = comoving_pos_idx % xfrac_file_high.mesh_x
		
		#Ionized fraction
		xi_interp = _get_interp_slice(xfrac_file_high.xi, xfrac_file_low.xi, z_bracket_high, \
									z_bracket_low, z, comoving_pos_idx)
		xfrac_lightcone[:,:,comoving_pos_idx] = xi_interp

		#Density
		rho_interp = _get_interp_slice(dens_file_high.cgs_density, dens_file_low.cgs_density, z_bracket_high, \
									z_bracket_low, z, comoving_pos_idx)
		dens_lightcone[:,:,comoving_pos_idx] = rho_interp

		#Brightness temperature
		dt_interp = _get_interp_slice(dt_cube_high, dt_cube_low, z_bracket_high, \
									z_bracket_low, z, comoving_pos_idx)
		dt_lightcone[:,:,comoving_pos_idx] = dt_interp

		print_msg( 'Slice %d of %d' % (comoving_pos_idx, len(output_z)) )
		comoving_pos_idx += 1

	return xfrac_lightcone, dens_lightcone, dt_lightcone, output_z


def make_lightcone(filenames, z_low = None, z_high = None, file_redshifts = None, \
				cbin_bits = 32, cbin_order = 'c'):
	'''
	Make a lightcone from xfrac, density or dT data.
	
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
		
	Returns:
		(lightcone, z) tuple
		lightcone is the lightcone volume where the first two axes
		have the same size as the input cubes
		
		z is an array containing the redshifts along the line-of-sight
	'''
	
	filenames = _get_filenames(filenames)
	file_redshifts = _get_file_redshifts(file_redshifts, filenames)
	assert(len(file_redshifts) == len(filenames))
	mesh_size = get_mesh_size(filenames[0])
		
	output_z = redshifts_at_equal_comoving_distance(z_low, z_high, box_grid_n=mesh_size[0])
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
									z_bracket_low, z, comoving_pos_idx)
		lightcone[:,:,comoving_pos_idx] = data_interp
		
		comoving_pos_idx += 1
		
	return lightcone, output_z



def _get_interp_slice(data_high, data_low, z_bracket_high, z_bracket_low, z, comoving_pos_idx):
	slice_ind = comoving_pos_idx % data_low.shape[0]
	slice_low = data_low[slice_ind,:,:]
	slice_high = data_high[slice_ind,:,:]
	slice_interp = ((z-z_bracket_low)*slice_high + (z_bracket_high - z)*slice_low)/(z_bracket_high-z_bracket_low)
	
	return slice_interp
	
	
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


#TEST--------------

if __name__ == '__main__':
	print _get_filenames('/home/hjens/links/local/slask/filenames_tmp.dat')
	
