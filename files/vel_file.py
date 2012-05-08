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
		import struct
		from scipy.io.numpyio import fread

		read_int = lambda f: struct.unpack('i', f.read(4))[0] #The format may be 'l' on some platforms
		file = open(filename, 'rb')
		self.mesh_x = read_int(file)
		self.mesh_y = read_int(file)
		self.mesh_z = read_int(file)

		self.raw_velocity = fread(file, self.mesh_x*self.mesh_y*self.mesh_z*3, 'f').astype('float64')
		file.close()
		self.raw_velocity = self.raw_velocity.reshape((3, self.mesh_x, self.mesh_y, self.mesh_z), order='F')

		#Store the redshift from the filename
		import os.path
		name = os.path.split(filename)[1]
		self.z = float(name.split('v_')[0])

		#Convert to kms/s*(rho/8)
		self.kmsrho8 = self.raw_velocity*conv.velconvert(z = self.z)

		#Store the redshift from the filename
		try:
			import os.path
			name = os.path.split(filename)[1]
			self.z = float(name.split('v_')[0])
		except:
			utils.print_msg('Could not determine redshift from file name')
			z = -1

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
