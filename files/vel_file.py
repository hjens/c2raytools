#This file defines the class VelocityFile which is used to read a file containing velocity data
#Useful attributes of the class are:
#raw_velocity -- the velocity in simulation units
#z -- the redshift of the file (-1 if it couldn't be determined from the file name
#To get the velocity in km/s, use get_kms_from_density

import numpy as np
from .. import const
from .. import conv
from .. import utils
from density_file import *

class VelocityFile:
	def __init__(self, filename = None):
		if filename:
			self.read_from_file(filename)

	def read_from_file(self, filename):
		utils.print_msg('Reading velocity file: %s...' % filename)

		#Read raw data from velocity file
		f = open(filename, 'rb')
		temp_mesh = np.fromfile(f, count=3, dtype='int32')
		self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh
		self.raw_velocity = np.fromfile(f, dtype='float32').astype('float64')
		f.close()
		self.raw_velocity = self.raw_velocity.reshape((3, self.mesh_x, self.mesh_y, self.mesh_z), order='F')

		#Store the redshift from the filename
		try:
			import os.path
			name = os.path.split(filename)[1]
			self.z = float(name.split('v_')[0])
		except:
			utils.print_msg('Could not determine redshift from file name')
			z = -1

		#Convert to kms/s*(rho/8)
		self.kmsrho8 = self.raw_velocity*conv.velconvert(z = self.z)


		utils.print_msg('...done')

	def get_kms_from_density(self, density):
		''' Get the velocity in kms. Since the file stores
		momentum rather than velocity, we need the density for this.
		density can be a string with a filename, a DensityFile object 
		or a numpy array with the raw density'''

		if isinstance(density,str):
			import density_file as df
			dfile = df.DensityFile(density)
			density = dfile.raw_density
		elif isinstance(density,DensityFile):
			density = density.raw_density

		return self.kmsrho8/(density/8)



#-------------------------TEST------------------------------
if __name__ == '__main__':
	velfile = VelocityFile(example_velocity)	
	print velfile.mesh_x, velfile.mesh_y, velfile.mesh_z
	print velfile.raw_velocity[0,0,0]
