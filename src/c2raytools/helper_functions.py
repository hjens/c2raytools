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

	import glob
	import os.path
	xfrac_files = glob.glob(os.path.join(xfrac_dir,'xfrac*.bin'))

	redshifts = []
	for f in xfrac_files:
		try:
			z = float(f.split('_')[-1][:-4])
			redshifts.append(z)
		except: 
			pass

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

	import glob
	import os.path
	dens_files = glob.glob(os.path.join(dens_dir,'*n_all.dat'))

	redshifts = []
	for f in dens_files:
		try:
			z = float(os.path.split(f)[1].split('n_')[0])
			redshifts.append(z)
		except:
			pass
		
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


def read_binary_with_meshinfo(filename, bits=32, order='C'):
	''' Read a binary file with three inital integers (a cbin file).
	
	Parameters:
		* filename (string): the filename to read from
		* bits = 32 (integer): the number of bits in the file
		* order = 'C' (string): the ordering of the data. Can be 'C'
			for C style ordering, or 'F' for fortran style.
			
	Returns:
		The data as a three dimensional numpy array.
	'''

	assert(bits ==32 or bits==64)

	f = open(filename)
	
	print_msg('Reading cbin file: %s' % filename)

	temp_mesh = np.fromfile(f,count=3,dtype='int32')
	mesh_x, mesh_y, mesh_z = temp_mesh

	datatype = np.float32 if bits == 32 else np.float64
	data = np.fromfile(f, dtype=datatype,count=mesh_x*mesh_y*mesh_z)
	data = data.reshape((mesh_x, mesh_y, mesh_z), order=order)
	return data


def read_raw_binary(filename, bits=64, order='C'):
	''' Read a raw binary file with no mesh info. The mesh
	is assumed to be cubic.
	
	Parameters:
		* filename (string): the filename to read from
		* bits = 64 (integer): the number of bits in the file
		* order = 'C' (string): the ordering of the data. Can be 'C'
			for C style ordering, or 'F' for fortran style.
			
	Returns:
		The data as a three dimensional numpy array.
	'''

	assert(bits ==32 or bits==64)

	f = open(filename)

	datatype = np.float32 if bits == 32 else np.float64
	data = np.fromfile(f, dtype=datatype)
	n = round(len(data)**(1./3.))
	data = data.reshape((n, n, n), order=order)
	return data


def save_binary_with_meshinfo(filename, data, bits=32, order='C'):
	''' Save a binary file with three inital integers (a cbin file).
	
	Parameters:
		* filename (string): the filename to save to
		* data (numpy array): the data to save
		* bits = 32 (integer): the number of bits in the file
		* order = 'C' (string): the ordering of the data. Can be 'C'
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
	
	
def read_fits(filename):
	'''
	Read a fits file and return the data as a numpy array
	
	Parameters:
		* filename (string): the fits file to read
		
	Returns:
		numpy array containing the data
	'''
	
	import pyfits as pf
	
	return pf.open(filename)[0].data.astype('float64')


def save_fits(data, filename):
	'''
	Save data as a fits file. The data can be a file object,
	a file to read or a pure data array.
	
	Parameters:
		* indata (XfracFile, DensityFile, string or numpy array): the data to save
		* filename (string): the file to save to
		
	Returns:
		Nothing
	
	'''
	import pyfits as pf
	
	save_data, datatype = get_data_and_type(data)
	
	hdu = pf.PrimaryHDU(save_data.astype('float64'))
	hdulist = pf.HDUList([hdu])
	hdulist.writeto(filename, clobber=True)
	

def determine_filetype(filename):
	'''
	Try to figure out what type of data is in filename.
	
	Parameters:
		* filename (string): the filename. May include the full
			path.
		
	Returns:
		A string with the data type. Possible values are:
		'xfrac', 'density', 'velocity', 'cbin', 'unknown'
		
	'''
	
	filename = os.path.basename(filename)
	
	if 'xfrac3d' in filename:
		return 'xfrac'
	elif 'n_all' in filename:
		return 'density'
	elif 'v_all' in filename:
		return 'velocity'
	elif '.cbin' in filename:
		return 'cbin'
	return 'unknown'


def get_data_and_type(indata, cbin_bits=32, cbin_order='c'):
	'''
	Extract the actual data from an object (which may
	be a file object or a filename to be read), and
	determine what type of data it is.
	
	Parameters:
		* indata (XfracFile, DensityFile, string or numpy array): the data
		* cbin_bits (integer): the number of bits to use if indata is a cbin file
		* cbin_order (string): the order of the data in indata if it's a cbin file
		
	Returns:
		* A tuple with (outdata, type), where outdata is a numpy array 
		containing the actual data and type is a string with the type 
		of data. Possible values for type are 'xfrac', 'density', 'cbin'
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
		elif filetype == 'cbin':
			return read_binary_with_meshinfo(indata, bits=cbin_bits, \
											order=cbin_order), 'cbin'
		else:
			raise Exception('Unknown file type')
	elif isinstance(indata, np.ndarray):
		return indata, 'unknown'
	raise Exception('Could not determine type of data')


def get_mesh_size(filename):
	'''
	Read only the first three integers that specify the mesh size of a file.
	
	Parameters:
		* filename (string): the file to read from. can be an xfrac file,
			a density file or a cbin file.
			
	Returns:
		(mx,my,mz) tuple
	'''
	datatype = determine_filetype(filename)
	f = open(filename, 'rb')
	if datatype == 'xfrac':
		temp_mesh = np.fromfile(f, count=6, dtype='int32')
		mesh_size = temp_mesh[1:4]
	elif datatype == 'density':
		mesh_size = np.fromfile(f,count=3,dtype='int32')
	elif datatype == 'cbin':
		mesh_size = np.fromfile(f,count=3,dtype='int32')
	else:
		raise Exception('Could not determine mesh for filetype %s' % datatype)
	f.close()
	return mesh_size
		

def outputify(output):
	'''
	If given a list with only one element, return the element
	If given a standard python list or tuple, make it into
	a numpy array.
	
	Parameters:
		output (any scalar or list-like): the output to process
		
	Returns:
		The output in the correct format.
	'''
	
	if hasattr(output, '__iter__'): #List-like
		if len(output) == 1:
			return output[0]
		elif not type(output) == np.ndarray:
			return np.array(output)
	return output
		
		
def determine_redshift_from_filename(filename):
	'''
	Try to find the redshift hidden in the filename.
	If there are many sequences of numbers in the filename
	this method will guess that the longest sequence is the
	redshift.
	
	Parameters:
		* filename (string): the filename to analyze
		
	Returns:
		* redshift (float)
	'''
	filename = os.path.basename(filename)
	filename = os.path.splitext(filename)[0]
	
	number_strs = [] #Will contain all sequences of numbers
	last_was_char = True
	for s in filename:
		if s.isdigit() or s == '.':
			if last_was_char:
				number_strs.append([])
			last_was_char = False
			number_strs[-1].append(s)
		else:
			last_was_char = True
	
	longest_idx = 0
	for i in range(len(number_strs)):
		if len(number_strs[i]) > len(number_strs[longest_idx]):
			longest_idx = i
		number_strs[i] = ''.join(number_strs[i])
		
	return float(number_strs[longest_idx])
			

verbose = False
def set_verbose(_verbose):
	'''
	Turn on or off verbose mode.
	
	Parameters:
		* verb (bool): whether or not to be verbose
		
	Returns:
		Nothing
	'''
	global verbose
	verbose = _verbose
