import numpy as np
import const
from helper_functions import print_msg
from xfrac_file import XfracFile
from density_file import DensityFile

def calc_dt(xfrac, dens, z = -1):
	'''
	Calculate the differential brightness temperature assuming T_s >> T_CMB
	
	Parameters:
		* xfrac (XfracFile object, string or numpy array): the ionization fraction
		* dens (DensityFile object, string or numpy array): density in sim units
		* z = -1 (float): The redshift (if < 0 this will be figured out from the files)
		
	Returns:
		The differential brightness temperature as a numpy array with
		the same dimensions as xfrac.
	'''

	#Figure out types of xfrac and dens
	if isinstance(xfrac, XfracFile):
		z = xfrac.z
		xi = xfrac.xi.astype('float64')
	elif isinstance(xfrac, str):
		xfile = XfracFile(xfrac)
		z = xfile.z
		xi = xfile.xi.astype('float64')
		if z < 0:
			print 'Warning. Please supply a redshift for calc_dt'
	else:
		xi = xfrac.astype('float64')

	if isinstance(dens, DensityFile):
		rho = dens.raw_density.astype('float64')
	elif isinstance(dens, str):
		dfile = DensityFile(dens)
		rho = dfile.raw_density.astype('float64')
	else:
		rho = dens.astype('float64')

	print_msg('Calculating differential brightness temperature...')
	print_msg('The redshift is %.3f' % z)

	rho_mean = np.mean(rho)

	#Redshift dependent Hubble constant
	Ez = np.sqrt(const.Omega0*(1.0+z)**3+const.lam+(1.0-const.Omega0-const.lam)*(1.0+z)**2)

	#The temperature box
	Cdt = const.meandt/const.h*(1.0+z)**2/Ez
	dt = Cdt*(1.0-xi)*rho/rho_mean
	
	print_msg('...done')

	return dt

#---------TEST-----------
if __name__ == '__main__':
	import c2raytools as c2t
	
	base_path = '/disk/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5' 
	density_filename = base_path+'/coarser_densities/nc256_halos_removed/8.515n_all.dat'
	xfrac_filename = base_path + '/114Mpc_f2_10S_256/results_ranger/xfrac3d_8.515.bin'

	c2t.set_verbose(True)
	xfile = c2t.XfracFile(xfrac_filename)
	dfile = c2t.DensityFile(density_filename)
	dT = c2t.calc_dt(xfile, dfile)
	
