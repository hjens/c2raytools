#This file contains various useful statistical methods

import numpy as np

def skewness(x):
	''' 
	Calculate the skewness of x
	IDL calculates the skewness in a slightly different way than Python. This routine uses the IDL definition 
	'''
        mx = mean(x)
        n = size(x)
        xdiff = x-mx
        #return (sum(xdiff**3)/n)/((sum(xdiff**2)/n)**(3./2.)) #This is how SciPy does it
   	return (sum(xdiff**3)/n)/((sum(xdiff**2)/(n-1))**(3./2.))


def mass_weighted_mean_xi(xi, rho):
	''' Calculate the mass-weighted mean ionization fraction '''
	xi = xi.astype('float64')
	rho = rho.astype('float64')
	return np.mean(xi*rho)/np.mean(rho)

                                                               
#TODO
#def kurtosis(x):

