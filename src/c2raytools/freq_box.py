import const
import conv
import numpy as np
import os
from temperature import calc_dt
from helper_functions import print_msg, get_dens_redshifts
from xfrac_file import XfracFile
from density_file import DensityFile


def freq_axis(z_low, z_high, box_length_slices=256, \
			box_length_mpc = conv.LB):
	''' 
	Make a frequency axis vector with equal spacing in co-moving LOS coordinates. 
	
	Parameters:
		* z_low (float): The lower redshift
		* z_high (float): The upper redhisft 
		* box_length_slices = 256 (int): the number of slices in an input box
		* box_length_mpc = conv.LB (float): the size of the box in cMpc
			 
	Returns:
		A tuple where the first element is a numpy array with the redshifts 
		and the second elemet is a numpy array with the corresponding 21 cm line 
		frequencies.
		
	'''
	assert(z_high > z_low)

	z = z_low
	z_array = []
	nu_array = []

	while z < z_high:
		nu = const.nu0/(1.0+z)

		z_array.append(z)
		nu_array.append(nu)

		dnu = const.nu0*const.Hz(z)*box_length_mpc/(1.0 + z)**2/const.c/float(box_length_slices)

		z = const.nu0/(nu - dnu) - 1.0

	return np.array(z_array), np.array(nu_array) 


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

	#Get the list of redhifts and frequencies that we want for the observational box
	output_z, output_freq = freq_axis(z_low, z_high)
	output_z = output_z[output_z > dens_redshifts[0]]
	output_z = output_z[output_z < dens_redshifts[-1]]
	if len(output_z) < 1:
		raise Exception('No valid redshifts in range!')

	print_msg( 'Number of slices reduced to: %d' % len(output_z) )

	#Keep track of output simulation files to use
	xfrac_file_low = XfracFile(); xfrac_file_high = XfracFile()
	dens_file_low = DensityFile(); dens_file_high = DensityFile()
	z_bracket_low = None; z_bracket_high = None

	#The current position in comoving coordinates
	nx = 0

	#Build the cube
	xfrac_cube = None
	dens_cube = None
	dt_cube = None
	for z in output_z:
		print_msg('z=%.3f' % z)
		#Find the output files that bracket the redshift
		z_bracket_low_new = dens_redshifts[np.where(dens_redshifts <= z)[0][0]]
		z_bracket_high_new = dens_redshifts[np.where(dens_redshifts >= z)[0][0]]

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

		slice_ind = nx % xfrac_file_high.mesh_x
		#The ionized fraction
		XL = xfrac_file_low.xi[slice_ind,:,:]
		XH = xfrac_file_high.xi[slice_ind,:,:]
		Xz = ((z-z_bracket_low)*XH + (z_bracket_high - z)*XL)/(z_bracket_high-z_bracket_low) #Interpolate between slices
		if xfrac_cube == None:
			xfrac_cube = np.zeros((XL.shape[0], XL.shape[1], len(output_z)))
		xfrac_cube[:,:,nx] = Xz

		#The density
		rho_L = dens_file_low.cgs_density[slice_ind,:,:]
		rho_H = dens_file_high.cgs_density[slice_ind,:,:]
		rho_Z = ((z-z_bracket_low)*rho_H + (z_bracket_high - z)*rho_L)/(z_bracket_high-z_bracket_low) #Interpolate between slices
		if dens_cube == None:
			dens_cube = np.zeros((XL.shape[0], XL.shape[1], len(output_z)))
		dens_cube[:,:,nx] = rho_Z

		#The brightness temperature
		dt_L = dt_cube_low[slice_ind,:,:]
		dt_H = dt_cube_high[slice_ind,:,:]
		dt_Z = ((z-z_bracket_low)*dt_H + (z_bracket_high - z)*dt_L)/(z_bracket_high-z_bracket_low) #Interpolate between slices
		if dt_cube == None:
			dt_cube = np.zeros((XL.shape[0], XL.shape[1], len(output_z)))
		dt_cube[:,:,nx] = dt_Z

		print_msg( 'Slice %d of %d' % (nx, len(output_z)) )
		nx += 1

	return xfrac_cube, dens_cube, dt_cube, output_z



