#This file contains various helper routines

import numpy as np
from scipy.interpolate import interp1d
from .. import const


def get_xfrac_redshifts(xfrac_dir, min_z = None, max_z = None):
	''' Make a list of xfrac files in xfrac_dir with redshifts between min_z and max_z 
	Return the redhifts of the files '''

	#Make a list of xfrac files to be read, in the correct redshift interval
	import glob
	import os.path
	xfrac_files = glob.glob(os.path.join(xfrac_dir,'xfrac*.bin'))

	#Get all z in the range
	redshifts = []
	for f in xfrac_files:
		try:
			z = float(f.split('_')[-1][:-4])
			redshifts.append(z)
		except: 
			pass
	#redshifts = [float(f.split('_')[-1][:-4]) for f in xfrac_files]
	if min_z:
		redshifts = filter(lambda x : x > min_z, redshifts)
	if max_z:
		redshifts = filter(lambda x : x < max_z, redshifts)
	redshifts.sort()

	return redshifts

def get_dens_redshifts(dens_dir, min_z = None, max_z = None):
	''' Make a list of density files in density_dir with redshifts between min_z and max_z 
	Return the redhifts of the files '''

	#Make a list of xfrac files to be read, in the correct redshift interval
	import glob
	import os.path
	dens_files = glob.glob(os.path.join(dens_dir,'*n_all.dat'))

	#Get all z in the range
	redshifts = []
	for f in dens_files:
		try:
			z = float(os.path.split(f)[1].split('n_')[0])
			redshifts.append(z)
		except:
			pass
	#redshifts = [float(os.path.split(f)[1].split('n_')[0]) for f in dens_files]
	if min_z:
		redshifts = filter(lambda x : x > min_z, redshifts)
	if max_z:
		redshifts = filter(lambda x : x < max_z, redshifts)
	redshifts.sort()

	return redshifts

def print_msg(message):
	''' Print a message of verbose is true '''
	if verbose:
		print message

def flt_comp(x,y, epsilon=0.0001):
	''' Compare two floats, return true of difference is < epsilon '''
	return abs(x-y) < epsilon

def get_interpolated_array(in_array, new_len, kind='nearest'):
	''' Get a higher-res version of in_array 
	new_len gives the length of the new array, kind gives the
	type of interpolation to perform ''' 

	old_len = len(in_array)
	func = interp1d(np.linspace(0,1,old_len), in_array, kind=kind)
	out_array = func(np.linspace(0,1,new_len))
	return out_array

def read_binary_with_meshinfo(filename, bits=32, order='F'):
	''' Read a binary file with three inital integers '''

	#import struct
	#from scipy.io.numpyio import fread

	assert(bits ==32 or bits==64)

	#read_int = lambda f: struct.unpack('i', f.read(4))[0] #The format may be 'l' on some platforms
	f = open(filename)

	temp_mesh = np.fromfile(f,count=3,dtype='int32')
	mesh_x, mesh_y, mesh_z = temp_mesh

	#data = fread(f, mesh_x*mesh_y*mesh_z,'f')
	datatype = dtype=(bits==32 and np.float32 or np.float64)
	data = np.fromfile(f, dtype=datatype,count=mesh_x*mesh_y*mesh_z)
	data = data.reshape((mesh_x, mesh_y, mesh_z), order=order)
	return data.astype('float64')

verbose = False
def set_verbose(verb):
	global verbose
	verbose = verb
