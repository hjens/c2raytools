#This file contains various useful statistical methods

import numpy as np

def skewness(x):
	''' 
	Calculate the skewness of an array.
	Note that IDL calculates the skewness in a slightly different way than Python. 
	This routine uses the IDL definition. 
	
	Parameters:
		* x (numpy array): The array containing the input data
		
	Returns:
		The skewness.
	
	'''
        mx = np.mean(x)
        n = np.size(x)
        xdiff = x-mx
        #return (sum(xdiff**3)/n)/((sum(xdiff**2)/n)**(3./2.)) #This is how SciPy does it
   	return (np.sum(xdiff**3)/n)/((sum(xdiff**2)/(n-1))**(3./2.))


def mass_weighted_mean_xi(xi, rho):
	''' Calculate the mass-weighted mean ionization fraction.
	
	Parameters:
		* xi (numpy array): the ionized fraction
		* rho (numpy array): the density (arbitrary units)
		
	Returns:
		The mean mass-weighted ionized fraction.
	
	 '''
	xi = xi.astype('float64')
	rho = rho.astype('float64')
	return np.mean(xi*rho)/np.mean(rho)

                                                               
#TODO
#def kurtosis(x):

