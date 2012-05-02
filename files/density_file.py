#This file defines the class DensityFile which is used to read a file containing density data
#Useful attributes of the class are:
#raw_density -- the density in simulation units
#cgs_density -- the density in cgs units
#z -- the redshift of the file (-1 if it couldn't be determined from the file name


import numpy as np
from .. import const
from .. import conv
from .. import utils 

class DensityFile:
	def __init__(self, filename = None, old_format = False):
		if filename:
			self.read_from_file(filename, old_format)

	def read_from_file(self, filename, old_format = False):
		''' Read density from file.
		If old_format is True, the file is assumed to contain no header data;
		the size is then taken to be 203**3 '''

		utils.print_msg('Reading density file:%s ...' % filename)
		#Read raw data from density file
		import struct
		from scipy.io.numpyio import fread
		f = open(filename, 'rb')

		if old_format:
			self.mesh_x = 203
			self.mesh_y = 203
			self.mesh_z = 203
		else:
			read_int = lambda f: struct.unpack('i', f.read(4))[0] #The format may be 'l' on some platforms
			self.mesh_x = read_int(f)
			self.mesh_y = read_int(f)
			self.mesh_z = read_int(f)

		self.raw_density = fread(f, self.mesh_x*self.mesh_y*self.mesh_z, 'f')
		self.raw_density = self.raw_density.reshape((self.mesh_x, self.mesh_y, self.mesh_z), order='F')
		
		f.close()

		#Convert to g/cm^3 (comoving)
		conv_factor = const.rho_crit_0*(float(self.mesh_x)/float(conv.nbox_fine))**3*const.OmegaB
		self.cgs_density = self.raw_density*conv_factor
		utils.print_msg('Mean density: %g' % np.mean(self.cgs_density))
		utils.print_msg('Critical density: %g' % const.rho_crit_0)

		#Store the redshift from the filename
		try:
			import os.path
			name = os.path.split(filename)[1]
			if old_format:
				self.z = float(name[:5])
			else:
				self.z = float(name.split('n_')[0])
		except:
			utils.print_msg('Could not determine redshift from file name')
			z = -1
		utils.print_msg( '...done')


