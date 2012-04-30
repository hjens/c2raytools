#This file defines the classes Halo and HaloList
#Halo is a simple class that holds som properties of a halo (explained in the class declaration)
#Halo list reads an ascii file with halo information and contains a list called halos

import numpy as np
from .. import utils
from .. import const
from .. import conv

#A simple struct to hold info about single halo
class Halo:
	def __init__(self):
		self.pos = (0.0, 0.0, 0.0) #Position in grid points
		self.pos_cm = (0.0, 0.0, 0.0) #Center of mass position in grid points
		self.vel = (0.0, 0.0, 0.0) #Velocity in simulation units
		self.l = (0.0, 0.0, 0.0) #Angular momentum in simulation units
		self.vel_disp = 0.0 #Velocity dispersion in simulation units
		self.r = 0.0 #Virial radius in grid units
		self.m = 0.0 #Grid mass
		self.mp = 0 #Number of particles
		self.solar_masses = 0.0 #Mass in solar masses

#Holds information about a large number of halos, as read from a halo list file
#Contains methods to select halos based on different criteria
class HaloList:
	def __init__(self, filename=None, min_select_mass = 0.0, max_select_mass = None, 
			max_select_number=-1, startline = 0):
		self.halos = []

		if filename:
			self.read_from_file(filename, min_select_mass, max_select_mass, max_select_number, 
					startline)

	def read_from_file(self,filename, min_select_mass = 0.0, max_select_mass = None, max_select_number=-1, 
			startline=0):
		''' Read a halo list from filename.
		Store halos with a mass larger than min_select_mass (solar masses)
		If old is True, assume the old file format
		Return True if all the halos were read '''

		self.halos = []

		utils.print_msg('Reading halo file %s...' % filename)
		import fileinput

		#Store the redshift from the filename
		import os.path
		name = os.path.split(filename)[1]
		self.z = float(name.split('halo')[0])

		#Read the file line by line, since it's large
		linenumber = 1
		min_select_grid_mass = min_select_mass/(conv.M_grid*const.solar_masses_per_gram)
		if max_select_mass:
			utils.print_msg('Max_select_mass: %g' % max_select_mass)
			max_select_grid_mass = max_select_mass/(conv.M_grid*const.solar_masses_per_gram)

		for line in fileinput.input(filename):
			if linenumber < startline: #If you want to read from a particular line
				linenumber += 1
				continue
			if max_select_number >= 0 and len(self.halos) >= max_select_number:
				fileinput.close()
				return False
			if linenumber % 100000 == 0:
				utils.print_msg('Read %d lines' % linenumber)
			linenumber += 1

			vals = line.split()
			grid_mass = float(vals[-3])

			#Create a halo and add it to the list
			if grid_mass > min_select_grid_mass and (max_select_mass == None or grid_mass < max_select_grid_mass):
				halo = Halo()
				halo.pos = np.array(map(float, vals[:3]))
				halo.pos_cm = np.array(map(float, vals[3:6]))
				halo.vel = np.array(map(float, vals[6:9]))
				halo.l = np.array(map(float, vals[9:12]))
				halo.vel_disp = float(vals[12])
				halo.r = float(vals[13])
				halo.m = float(vals[14])
				halo.mp = float(vals[15])
				halo.solar_masses = grid_mass*conv.M_grid*const.solar_masses_per_gram
				self.halos.append(halo)

		fileinput.close()

		return True

			

