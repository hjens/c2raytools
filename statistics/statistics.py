from numpy import *

def idlskew(x):
	''' IDL calculates the skewness in a slightly different way than Python. This routine uses the IDL definition '''
        mx = mean(x)
        n = size(x)
        xdiff = x-mx
        #return (sum(xdiff**3)/n)/((sum(xdiff**2)/n)**(3./2.)) #This is how SciPy does it
        return (sum(xdiff**3)/n)/((sum(xdiff**2)/(n-1))**(3./2.))
                                                               

def basic_stats(box):
	return (mean(box), std(box), idlskew(box))

def dt_stats(xfrac_file, density_file):
	dt = xfrac_file.calc_dt(density_file)
	return basic_stats(dt)

def dt_histogram(xfrac_file, density_file):
	dt = xfrac_file.calc_dt(density_file)
	return histogram(dt.flat, bins=linspace(0,80,30), normed=True)[0]

def basic_stats_for_dir(xfrac_dir, density_dir, min_z=-1, max_z=-1):
	
	from helper_functions import run_func_for_filenames
	return run_func_for_filenames(xfrac_dir, density_dir, dt_stats, min_z, max_z)

def redshift_dt_histogram(xfrac_dir, density_dir, min_z=-1, max_z=-1):
	from helper_functions import run_func_for_filenames
	return run_func_for_filenames(xfrac_dir, density_dir, dt_histogram, min_z, max_z)
	

#-------------TEST----------------------
if __name__ == '__main__':
	from xfrac_file import * 
	from pylab import *

	#hist_surf, z = redshift_dt_histogram(xfrac_dir = '114Mpc_WMAP5/114Mpc_f10_150S_256/results/', density_dir='114Mpc_WMAP5/coarser_densities/halos_removed')
	#temp = open('hist_surf.txt', 'w')
	#savetxt(temp, array(hist_surf))
	#temp.close()
	#temp = open('hist_z.txt', 'w')
	#savetxt(temp, array(z))
	#temp.close()

	xfrac_dir = '/export/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5/114Mpc_f10_150S_256/results'
	density_dir = '/export/sn-12/garrelt/Science/Simulations/Reionization/C2Ray_WMAP5/114Mpc_WMAP5/coarser_densities/halos_removed'
	stats, z = basic_stats_for_dir(xfrac_dir = xfrac_dir, density_dir=density_dir)
	dt_mean = array([f[0] for f in stats])
	dt_std = array([f[1] for f in stats])
	dt_skew = array([f[2] for f in stats])

	figure()
	plot(z,dt_mean)
	xlabel('z')
	ylabel('Mean $T_b$')

	figure()
	plot(z,dt_std)
	xlabel('z')
	ylabel('std $T_b$')

	figure()
	plot(z,dt_skew)
	xlabel('z')
	ylabel('skew $T_b$')

	show()
#-------------TEST----------------------
