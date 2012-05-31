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

		f = open(filename, 'rb')
		temp_mesh = np.fromfile(f, count=6, dtype='int32')
		self.mesh_x, self.mesh_y, self.mesh_z = temp_mesh[1:4]

		if old_format:
			self.xi = np.fromfile(f, dtype='float32')
		else:
			self.xi = np.fromfile(f, dtype='float64')
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



#--------------------TEST-----------------------
if __name__ == '__main__':
	pass
#--------------------TEST-----------------------
