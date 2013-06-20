from .. import const
from .. import conv
import numpy as np
import os
from .. import files
from temperature import calc_dt
from .. import utils

def freq_axis(z_low, output_slices, box_length_slices=256, box_length_mpc = conv.LB):
	''' 
	Make a frequency axis vector with equal spacing in co-moving LOS coordinates. 
	
	Parameters:
		* z_low (float): The lowest redshift
		* output_slices (int): the number of slices in the output array
		* box_length_slices = 256 (int): the number of slices in an input box
		* box_length_mpc = conv.LB (float): the desired number of datapoints 
			on the axis
			 
	Returns:
		A tuple where the first element is a numpy array with the redshifts 
		and the second elemet is a numpy array with the corresponding 21 cm line 
		frequencies.
		
	TODO: 
		Add option to input z_high
	'''

	z = z_low
	z_array = np.zeros((output_slices,2))

	for nx in xrange(output_slices):
		nu = const.nu0/(1.0+z)

		z_array[nx,0] = z
		z_array[nx,1] = nu

		dnu = const.nu0*const.Hz(z)*box_length_mpc/(1.0 + z)**2/const.c/float(box_length_slices)

		z = const.nu0/(nu - dnu) - 1.0


	return z_array[:,0], z_array[:,1]


def freq_box(xfrac_dir, dens_dir, z_low, cube_slices=100):
	''' 
	Make frequency (lightcone) boxes of density, ionized fractions, 
	and brightness temperature. This routine is more or less a 
	direct translation of Garrelts IDL routine. I have not tested it
	much. Use at your own risk.
	
	Parameters: 
		* xfrac_dir (string): directory containing xfrac files
		* dens_dir (string): directory containing density files
		* z_low (float): lowest redshift to include
		* cube_slices = 100 (int): number of slices to divide the cube into

	Returns: 
		Tuple with (density box, xfrac box, dt box, redshifts), where
		density box, xfrac box and dt box are numpy arrays containing
		the lightcone quantities. redshifts is an array containing the 
		redshift for each slice.
		
	TODO:
		Test this routine. Make it more general. Check error messages.
	'''


	#Get the list of redshifts where we have simulation output files
	dens_redshifts = utils.get_dens_redshifts(dens_dir, z_low )

	#Get the list of redhifts and frequencies that we want for the observational box
	output_z, output_freq = freq_axis(z_low, cube_slices)
	output_z = np.delete(output_z, np.where(output_z < dens_redshifts[0])[0])
	output_z = np.delete(output_z, np.where(output_z > dens_redshifts[-1])[0])

	utils.print_msg( 'Number of slices reduced to: %d' % len(output_z) )

	#Keep track of output simulation files to use
	xfrac_filename_high = None; xfrac_filename_low = None
	xfrac_file_low = files.XfracFile(); xfrac_file_high = files.XfracFile()
	dens_filename_high = None; dens_filename_low = None
	dens_file_low = files.DensityFile(); dens_file_high = files.DensityFile()
	z_bracket_low = None; z_bracket_high = None

	#The current position in comoving coordinates
	nx = 0

	#Build the cube
	xfrac_cube = None
	dens_cube = None
	dt_cube = None
	for z in output_z:
		utils.print_msg('z=%.3f' % z)
		#Find the output files that bracket the redshift
		z_bracket_low_new = dens_redshifts[np.where(dens_redshifts <= z)[0][0]]
		z_bracket_high_new = dens_redshifts[np.where(dens_redshifts >= z)[0][0]]

		if z_bracket_low_new != z_bracket_low:
			z_bracket_low = z_bracket_low_new
			xfrac_file_low = files.XfracFile(os.path.join(xfrac_dir, 'xfrac3d_%.3f.bin' % z_bracket_low))
			dens_file_low = files.DensityFile(os.path.join(dens_dir, '%.3fn_all.dat' % z_bracket_low))
			dt_cube_low = calc_dt(xfrac_file_low, dens_file_low)

		if z_bracket_high_new != z_bracket_high:
			z_bracket_high = z_bracket_high_new
			xfrac_file_high = files.XfracFile(os.path.join(xfrac_dir, 'xfrac3d_%.3f.bin' % z_bracket_high))
			dens_file_high = files.DensityFile(os.path.join(dens_dir, '%.3fn_all.dat' % z_bracket_high))
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

		utils.print_msg( 'Slice %d of %d' % (nx, len(output_z)) )
		nx += 1

	return xfrac_cube, dens_cube, dt_cube, output_z




