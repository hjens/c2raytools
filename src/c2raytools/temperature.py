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
		* dens (DensityFile object, string or numpy array): density in cgs units
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

def calc_dt_full(xfrac, temp, dens, z = -1):
	'''
	Calculate the differential brightness temperature assuming only that Lyman alpha is fully coupled so T_s = T_k
    (NOT T_s >> T_CMB)
	
	Parameters:
		* xfrac (XfracFile object, string or numpy array): the ionization fraction
        * temp (TemperFile object, string or numpy array): the temperature in K
		* dens (DensityFile object, string or numpy array): density in cgs units
		* z = -1 (float): The redshift (if < 0 this will be figured out from the files)
		
	Returns:
		The differential brightness temperature as a numpy array with
		the same dimensions as xfrac.
	'''

	xi, xi_type = get_data_and_type(xfrac)
        Ts, Ts_type = get_data_and_type(temp)
	rho, rho_type = get_data_and_type(dens)
	xi = xi.astype('float64')
        Ts = Ts.astype('float64')
	rho = rho.astype('float64')
	
	if z < 0:
		z = determine_redshift_from_filename(xfrac)
		if z < 0:
			z = determine_redshift_from_filename(dens)
                        if z < 0:
                		z=determine_redshift_from_filename(temp)
				if z < 0:
					raise Exception('No redshift specified. Could not determine from file.')
	
	print_msg('Making full dT box for z=%f' % z)
	
	#Calculate dT
        print "calculating corrected dbt"
	return _dt_full_corrected(dens,xfrac,temp,z)#rho, Ts, xi, z)

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
	dens = dens.astype('float64')
		
	cell_size = conv.LB/xfrac.shape[(los_axis+1)%3]
	cdist_low = cosmology.z_to_cdist(lowest_z)
	cdist = np.arange(xfrac.shape[los_axis])*cell_size + cdist_low
	z = cosmology.cdist_to_z(cdist)
	return _dt(dens, xfrac, z)

def calc_dt_full_lightcone(xfrac, temp, dens, lowest_z, los_axis = 2):
	'''
	Calculate the differential brightness temperature assuming only that Lyman alpha is fully coupled so T_s = T_k
    (NOT T_s >> T_CMB) for lightcone data. UNTESTED
	
	Parameters:
		* xfrac (string or numpy array): the name of the ionization 
			fraction file (must be cbin), or the xfrac lightcone data
        * temp (string or numpy array): the name of the temperature
            file (must be cbin), or the temp lightcone data
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
		temp = read_cbin(temp)
	except Exception:
		pass
	try:
		dens = read_cbin(dens)
	except:
		pass
	dens = dens.astype('float64')
		
	cell_size = conv.LB/xfrac.shape[(los_axis+1)%3]
	cdist_low = cosmology.z_to_cdist(lowest_z)
	cdist = np.arange(xfrac.shape[los_axis])*cell_size + cdist_low
	z = cosmology.cdist_to_z(cdist)
        print "Redshift: ", str(z)
	return _dt_full_corrected(dens, xfrac,temp, z)

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
		
	rho_mean = const.rho_crit_0*const.OmegaB

	Cdt = mean_dt(z)
	dt = Cdt*(1.0-xi)*rho/rho_mean
	
	return dt

def _dt_full(rho, xi, Ts, z):
        print "Redshift: " +str(z) +'\n'
	rho_mean = const.rho_crit_0*const.OmegaB
        Tcmb = 2.725*(1+z) # might want to add to cosmology.py instead
	Cdt = mean_dt(z)
        print type(Ts),type(Tcmb),type(Cdt),type(xi),type(rho)
        dt = ((Ts.temper-Tcmb)/Ts.temper)*Cdt*(1.0-xi)*rho/rho_mean  #extra term for temperature fluctuations as Ts is not much greater than Tcmb, Hannah Ross
        #dt = ((Ts-Tcmb)/Ts)*Cdt*(1.0-xi.xi)*rho.cgs_density/rho_mean  #extra term for temperature fluctuations as Ts is not much greater than Tcmb, Hannah Ross
#	dt = (1+Tcmb/Ts)*Cdt*(1.0-xi)*rho/rho_mean
	
	return dt
	
def _dt_full_corrected(rho, xi, Ts, z):
        z = np.mean(z)
        print "Redshift: " +str(z) +'\n'
        rho_mean = const.rho_crit_0*const.OmegaB
        Tcmb = 2.725*(1+z) # might want to add to cosmology.py instead
        Cdt = mean_dt(z)

        zi = 1089
        Tcmb0=2.726
        Ti=Tcmb0*(1+zi)
        T_HII=2.0e4
        T_min = Ti*(1+z)**2/(1+zi)**2
#        print T_min
        ones=np.ones(250**3,dtype=float).reshape(250,250,250)
#        print z
        #print type(z),type(Ts.temper), type(T_min), type(T_HII), type(xi)
        for i in range(len(xi[:,1,1])):
            for j in range(len(xi[1,:,1])):
                for k in range(len(xi[1,1,:])):
                    Ts.temper[i,j,k] = max(T_min,(Ts.temper[i,j,k]-xi[i,j,k]*T_HII)/(1.-xi[i,j,k]))
                        
 #       print np.mean(Ts.temper)
#        Ts.temper = (Ts.temper-xi*T_HII)/(ones-xi)
        if np.any(Ts.temper < 0):
            print "WARNING: negative temperatures"
#        print type(Ts),type(Tcmb),type(Cdt),type(xi),type(rho)

        dt = ((Ts.temper-Tcmb)/Ts.temper)*Cdt*(1.0-xi)*rho/rho_mean  #extra term for temperature fluctuations as Ts is not much greater than Tcmb, Hannah Ross
        #dt = ((Ts-Tcmb)/Ts)*Cdt*(1.0-xi.xi)*rho.cgs_density/rho_mean  #extra term for temperature fluctuations as Ts is not much greater than Tcmb, Hannah Ross
#       dt = (1+Tcmb/Ts)*Cdt*(1.0-xi)*rho/rho_mean

        return dt

