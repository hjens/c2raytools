import numpy as np
from .. import const
from .. import conv
from .. import utils 

class DensityFile:
	'''
	A CubeP3M density file.
	
	Use the read_from_file method to load a density file, or 
	pass the filename to the constructor.
	
	Some useful attributes of this class are:
	raw_density -- the density in simulation units
	cgs_density -- the density in cgs units
	z -- the redshift of the file (-1 if it couldn't be determined from the file name)
	
	'''
	
	def __init__(self, filename = None, old_format = False):
		'''
		Initialize the file. If filename is given, read data. Otherwise,
		do nothing.
		Parameters:
			* filename = None (file object or filename): the file to 
			read from.
			* old_format = False (bool): whether to use the old-style 
			file format.
		Returns:
			Nothing
		'''
		if filename:
			self.read_from_file(filename, old_format)

	def read_from_file(self, filename, old_format = False):
		'''
		Read data from file.
		Parameters:
			* filename (file object or filename): the file to 
			read from.
			* old_format = False (bool): whether to use the old-style 
			file format.
		Returns:
			Nothing
		'''

		utils.print_msg('Reading density file:%s ...' % filename)
		#Read raw data from density file
		f = open(filename, 'rb')

		if old_format:
			self.mesh_x = 203
			self.mesh_y = 203
			self.mesh_z = 203
		else:
			temp_mesh = np.fromfile(f,count=3,dtype='int32')
			self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh

		self.raw_density = np.fromfile(f, dtype='float32')
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
			self.z = -1
		utils.print_msg( '...done')


