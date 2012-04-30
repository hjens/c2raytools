from cosm_constants import *
def tau(ionfractions, redshifts):

	if len(ionfractions) != len(redshifts):
		print 'Incorrect length of ionfractions'
		raise Exception()
	
	sigma_T=6.65e-25
	chi1=1.0+abu_he
	coeff = 2.0*(c*1e5)*sigma_T*OmegaB/Omega0*rho_crit_0*chi1/mean_molecular/m_p/(3.*H0cgs)

	num_points = 50

	tau_z = hstack((arange(1,num_points+1)/float(num_points)*redshifts[0], redshifts))

	tau0=zeros(len(redshifts)+num_points)
	tau0[0:num_points] = coeff*(sqrt(Omega0*(1+tau_z[0:num_points])**3+lam) - 1)

	for i in xrange(num_points, len(redshifts)+num_points):
		tau0[i] =tau0[i-1]+1.5*coeff*Omega0 * \
		(ionfractions[i-1-num_points]*(1+tau_z[i-1])**2/sqrt(Omega0*(1+tau_z[i-1])**3+lam) \
		+ ionfractions[i-num_points]*(1+tau_z[i])**2/sqrt(Omega0*(1+tau_z[i])**3+lam) ) * \
		(tau_z[i]-tau_z[i-1])/2


	return tau0, tau_z


#--------------------TEST-----------------------
if __name__ == '__main__':
	from xfrac_file import * 
	fracs,z = get_ionfracs_from_dir(xfrac_dir = '114Mpc_WMAP5/114Mpc_f10_150S_256/results/', density_dir='114Mpc_WMAP5/coarser_densities/halos_removed')
	iof_vol = array([f[0] for f in fracs])
	iof_mass = array([f[1] for f in fracs])

	temp = open('xfracs.txt', 'w')
	for frac in fracs:
		temp.write('%.4e %.4e %.4e %.4e \n' % frac)
	temp.close()

	tau0, tau_z = tau(iof_mass, z)
	print 'len(tau0)', len(tau0)
	print 'len(tau_z)', len(tau_z)
	temp = open('tau0.txt', 'w')
	savetxt(temp, tau0, delimiter='\n')
	temp.close()
	temp = open('tau_z.txt', 'w')
	savetxt(temp, tau_z, delimiter='\n')
	temp.close()

	from pylab import *
	#figure()
	#plot(tau_z, tau0, '+-')
	#xlabel('z')
	#ylabel('tau_0')
	#show()

#--------------------TEST-----------------------
	

  
