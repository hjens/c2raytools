#Various functions for calculating some cosmological stuff
from .. import const
import numpy as np
from scipy.integrate import quadrature

def ldist(z):
	'''This function is used for the integration in lumdist 
	Only meant for internal use '''
	term1 = (1+z)**2
	term2 =  1.+2.*(const.q0+const.lam)*z
	term3 = z*(2.+z)*const.lam
	denom = (term1*term2 - term3)
	if type(z) is np.ndarray:
		out = np.zeros(z.shape)
		good = np.where(denom > 0.0)[0]
		out[good] = 1.0/np.sqrt(denom[good])
		return out
	else:
		if denom >= 0:
			return 1.0/np.sqrt(denom)
		else:
			return 0.0


def lumdist(z, k=0):
	''' Calculate the luminosity distance for redshift z '''

	if not (type(z) is np.ndarray): #Allow for passing a single z
		z = np.array([z])
	n = len(z)

	if const.lam == 0:
		denom = np.sqrt(1+2*const.q0*z) + 1 + const.q0*z 
		dlum = (const.c*z/const.h0)*(1 + z*(1-const.q0)/denom)
		return dlum
	else:
		dlum = np.zeros(n)
		for i in xrange(n):
			if z[i] <= 0:
				dlum[i] = 0.0
			else:
				dlum[i] = quadrature(ldist, 0, z[i])[0]

	if k > 0:
		dlum = np.sinh(np.sqrt(k)*dlum)/np.sqrt(k)
	elif k < 0:
		dlum = np.sin(np.sqrt(-k)*dlum)/np.sqrt(-k)
	return const.c*(1+z)*dlum/const.H0


def zang(dl, z):
	''' Calculate the angular size of an object. 
	dl is the physical size in kpc
	z is the redshift of the object
	The result is given in arcseconds '''

	angle = 180./(3.1415)*3600.*dl*(1+z)**2/(1000*lumdist(z))
	if len(angle) == 1:
		return angle[0]
	return angle
