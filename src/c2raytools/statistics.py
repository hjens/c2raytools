#This file contains various useful statistical methods

import numpy as np
from lightcone import _get_slice

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
	return (np.sum(xdiff**3)/n)/((np.sum(xdiff**2)/(n-1))**(3./2.))


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


def subtract_mean_signal(signal, los_axis):
	'''
	Subtract the mean of the signal along the los axis. The
	mean is calculated for a number of slice along the los.
	
	Parameters:
		* signal (numpy array): the signal to subtract the mean from
		* los_axis (integer): the line-of-sight axis
		* slice_depth=1 (integer): the depth of each slice in which
			to calculate the mean
			
	Returns:
		The signal with the mean subtracted
		
	TODO:vectorize 
	'''
	signal_out = signal.copy()
	
	for i in range(signal.shape[los_axis]):
		if los_axis == 0:
			signal_out[i,:,:] -= signal[i,:,:].mean()
		if los_axis == 1:
			signal_out[:,i,:] -= signal[:,i,:].mean()
		if los_axis == 2:
			signal_out[:,:,i] -= signal[:,:,i].mean()

	return signal_out

                                                               
#TODO
#def kurtosis(x):

