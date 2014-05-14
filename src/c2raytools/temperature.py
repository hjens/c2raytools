import numpy as np
import const
import conv
import cosmology
from helper_functions import print_msg, read_cbin, \
	get_data_and_type, determine_redshift_from_filename

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

	xi, xi_type = get_data_and_type(xfrac)
	rho, rho_type = get_data_and_type(dens)
	xi = xi.astype('float64')
	rho = rho.astype('float64')
	
	if z < 0:
		z = determine_redshift_from_filename(xfrac)
		if z < 0:
			z = determine_redshift_from_filename(dens)
		if z < 0:
			raise Exception('No redshift specified. Could not determine from file.')
	
	print_msg('Making dT box for z=%f' % z)
	
	#Calculate dT
	return _dt(rho, xi, z)
	

def calc_dt_lightcone(xfrac, dens, lowest_z, los_axis = 2):
	'''
	Calculate the differential brightness temperature assuming T_s >> T_CMB
	for lightcone data.
	
	Parameters:
		* xfrac (string or numpy array): the name of the ionization 
			fraction file (must be cbin), or the xfrac lightcone data
		* dens (string or numpy array): the name of the density 
			file (must be cbin), or the density data
		* lowest_z (float): the lowest redshift of the lightcone volume
		* los_axis = 2 (int): the line-of-sight axis
		
	Returns:
		The differential brightness temperature as a numpy array with
		the same dimensions as xfrac.
	'''
	
	try:
		xfrac = read_cbin(xfrac)
	except Exception:
		pass
	try:
		dens = read_cbin(dens)
	except:
		pass
		
	cell_size = conv.LB/xfrac.shape[(los_axis+1)%3]
	cdist_low = cosmology.z_to_cdist(lowest_z)
	cdist = np.arange(xfrac.shape[los_axis])*cell_size + cdist_low
	z = cosmology.cdist_to_z(cdist)
	return _dt(dens, xfrac, z)


def mean_dt(z):
	'''
	Get the mean dT at redshift z
	
	Parameters:
		* z (float or numpy array): the redshift
		
	Returns:
		dT (float or numpy array) the mean brightness temperature
		in mK
	'''
	Ez = np.sqrt(const.Omega0*(1.0+z)**3+const.lam+\
				(1.0-const.Omega0-const.lam)*(1.0+z)**2)

	Cdt = const.meandt/const.h*(1.0+z)**2/Ez
	
	return Cdt
	

def _dt(rho, xi, z):
	rho_mean = np.mean(rho.astype('float64'))

	Cdt = mean_dt(z)
	dt = Cdt*(1.0-xi)*rho/rho_mean
	
	return dt
	

