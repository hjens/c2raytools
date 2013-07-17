#This file contains various helper routines

import numpy as np
from scipy.interpolate import interp1d
import const
import os



def get_xfrac_redshifts(xfrac_dir, min_z = None, max_z = None):
	''' 
	Make a list of the redshifts of all the xfrac files in a directory.
	
	Parameters:
		* xfrac_dir (string): the directory to look in
		* min_z = None (float): the minimum redshift to include (if given)
		* max_z = None (float): the maximum redshift to include (if given)
	 
	Returns: 
		The redhifts of the files (numpy array of floats) '''

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

	return np.array(redshifts)

def get_dens_redshifts(dens_dir, min_z = None, max_z = None):
	''' 
	Make a list of the redshifts of all the density files in a directory.
	
	Parameters:
		* dens_dir (string): the directory to look in
		* min_z = None (float): the minimum redshift to include (if given)
		* max_z = None (float): the maximum redshift to include (if given)
	 
	Returns: 
		The redhifts of the files (numpy array of floats) '''

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

	return np.array(redshifts)

def print_msg(message):
	''' Print a message of verbose is true '''
	if verbose:
		print message

def flt_comp(x,y, epsilon=0.0001):
	''' Compare two floats, return true of difference is < epsilon '''
	return abs(x-y) < epsilon

def get_interpolated_array(in_array, new_len, kind='nearest'):
	''' Get a higher-res version of an array.
	
	Parameters:
		* in_array (numpy array): the array to upscale
		* new_len (integer): the new length of the array
		* kind = 'nearest' (string): the type of interpolation to use
		
	Returns:
		The upscaled array. 
	''' 

	old_len = len(in_array)
	func = interp1d(np.linspace(0,1,old_len), in_array, kind=kind)
	out_array = func(np.linspace(0,1,new_len))
	return out_array

def read_binary_with_meshinfo(filename, bits=32, order='F'):
	''' Read a binary file with three inital integers (a cbin file).
	
	Parameters:
		* filename (string): the filename to read from
		* bits = 32 (integer): the number of bits in the file
		* order = 'F' (string): the ordering of the data. Can be 'C'
			for C style ordering, or 'F' for fortran style.
			
	Returns:
		The data as a three dimensional numpy array.
	'''

	#import struct
	#from scipy.io.numpyio import fread

	assert(bits ==32 or bits==64)

	#read_int = lambda f: struct.unpack('i', f.read(4))[0] #The format may be 'l' on some platforms
	f = open(filename)

	temp_mesh = np.fromfile(f,count=3,dtype='int32')
	mesh_x, mesh_y, mesh_z = temp_mesh

	#data = fread(f, mesh_x*mesh_y*mesh_z,'f')
	datatype = (bits==32 and np.float32 or np.float64)
	data = np.fromfile(f, dtype=datatype,count=mesh_x*mesh_y*mesh_z)
	data = data.reshape((mesh_x, mesh_y, mesh_z), order=order)
	return data.astype('float64')

def save_binary_with_meshinfo(filename, data, bits=32, order='F'):
	''' Save a binary file with three inital integers (a cbin file).
	
	Parameters:
		* filename (string): the filename to save to
		* data (numpy array): the data to save
		* bits = 32 (integer): the number of bits in the file
		* order = 'F' (string): the ordering of the data. Can be 'C'
			for C style ordering, or 'F' for fortran style.
			
	Returns:
		Nothing
	'''
	assert(bits ==32 or bits==64)
	f = open(filename, 'wb')
	mesh = np.array(data.shape).astype('int32')
	mesh.tofile(f)
	datatype = (np.float32 if bits==32 else np.float64)
	data.flatten(order=order).astype(datatype).tofile(f)
	f.close()

def determine_filetype(filename):
	'''
	Try to figure out what type of data is in filename.
	
	Parameters:
		* filename (string): the filename. May include the full
			path.
		
	Returns:
		A string with the data type. Possible values are:
		'xfrac', 'density', 'velocity', 'unknown'
		
	'''
	
	filename = os.path.basename(filename)
	
	if 'xfrac3d' in filename:
		return 'xfrac'
	elif 'n_all' in filename:
		return 'density'
	elif 'v_all' in filename:
		return 'velocity'
	return 'unknown'

def get_data_and_type(indata):
	'''
	Extract the actual data from an object (which may
	be a file object or a filename to be read), and
	determine what type of data it is.
	
	Parameters:
		* indata (XfracFile, DensityFile, string or numpy array): the data
		
	Returns:
		* A tuple with (outdata, type), where outdata is a numpy array 
		containing the actual data and type is a string with the type 
		of data. Possible values for type are 'xfrac', 'density', 
		and 'unknown'
		
	'''
	import c2raytools.density_file
	import c2raytools.xfrac_file

	if isinstance(indata, c2raytools.xfrac_file.XfracFile):
		return indata.xi, 'xfrac'
	elif isinstance(indata, c2raytools.density_file.DensityFile):
		return indata.cgs_density, 'density'
	elif isinstance(indata, str):
		filetype = determine_filetype(indata)
		if filetype == 'xfrac':
			return get_data_and_type(c2raytools.xfrac_file.XfracFile(indata))
		elif filetype == 'density':
			return get_data_and_type(c2raytools.density_file.DensityFile(indata))
		else:
			raise Exception('Unknown file type')
	elif isinstance(indata, np.ndarray):
		return indata, 'unknown'
	raise Exception('Could not determine type of data')
		

verbose = False
def set_verbose(verb):
	'''
	Turn on or off verbose mode.
	
	Parameters:
		* verb (bool): whether or not to be verbose
		
	Returns:
		Nothing
	'''
	global verbose
	verbose = verb
