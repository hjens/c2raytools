#This file defines the class XfracFile which is used to read a file containing ionization fractions
#Useful attributes of the class are:
#xi -- the ionized fraction
#z -- the redshift of the file (-1 if it couldn't be determined from the file name

from .. import const
import numpy as np
import density_file as df
from .. import utils 

class XfracFile:
	def __init__(self, filename = None, old_format=False):
		if filename:
			self.read_from_file(filename, old_format)

	def read_from_file(self, filename, old_format=False):
		''' Read the ionization fraction file
		If old_format is true, the precision is taken to be 32 bits '''
		utils.print_msg('Reading xfrac file:%s...' % filename)
		import struct
		from scipy.io.numpyio import fread

		read_int = lambda f: struct.unpack('i', f.read(4))[0] #The format may be 'l' on some platforms
		f = open(filename, 'rb')
		dummy = read_int(f)
		self.mesh_x = read_int(f)
		self.mesh_y = read_int(f)
		self.mesh_z = read_int(f)
		dummy = read_int(f)
		dummy = read_int(f)

		if old_format:
			self.xi = fread(f, self.mesh_x*self.mesh_y*self.mesh_z, 'f')
		else:
			self.xi = fread(f, self.mesh_x*self.mesh_y*self.mesh_z, 'd')
		self.xi = self.xi.reshape((self.mesh_x, self.mesh_y, self.mesh_z), order='F')

		f.close()
		utils.print_msg('...done')

		#Store the redshift from the filename
		import os.path
		try:
			name = os.path.split(filename)[1]
			self.z = float(name.split('_')[1][:-4])
		except:
			utils.print_msg('Could not determine redshift from file name')
			x = -1

	#def calc_mean_xfracs(self, dfile):
	#	''' Return a tuple with (mean xi by volume, mean xi by mass, 1-mean xi by vol, 1-mean xi by mass '''

	#	utils.print_msg('Calculating mean ionization fractions...')
	#	rho = dfile.raw_density.astype('float64')
	#	xi = self.xi.astype('float64')
	#	mean_rho = np.mean(rho)
	#	mean_xi = np.mean(xi)
	#	mean_xi_mass = np.mean(xi*rho)/mean_rho
	#	utils.print_msg('...done')
	#	return (mean_xi, mean_xi_mass, 1.0-mean_xi, np.mean((1.0-xi)*rho)/mean_rho)



#def get_ionfracs_from_dir(xfrac_dir, density_dir, min_z=-1, max_z = -1):
#	''' Calculate the mean ionization fractions as a function of redshift
#	using the files in xfrac_dir and density_dir '''
#
#	from helper_functions import run_func_for_filenames
#	callback = lambda f,d: f.calc_mean_xfracs(d)
#	return run_func_for_filenames(xfrac_dir, density_dir, callback, min_z, max_z)


#--------------------TEST-----------------------
if __name__ == '__main__':
	pass
#--------------------TEST-----------------------
