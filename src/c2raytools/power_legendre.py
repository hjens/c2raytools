'''
Created on Aug 21, 2013

@author: Hannes Jensen
'''

import numpy as np
from power_spectrum import _get_dims, power_spectrum_nd, _get_k, _get_kbins, _get_mu

def power_spectrum_multipoles(input_array, kbins = 10, box_dims = None, los_axis = 0):
    '''
    Calculate the power spectrum of an array and 
    expand it in the first three Legendre polynomials.
    
    Parameters:
        * input_array (numpy array): the array to calculate the 
            power spectrum of. Can be of any dimensions.
        * kbins = 10 (integer or array-like): The number of bins,
            or a list containing the bin edges. If an integer is given, the bins
            are logarithmically spaced.
        * box_dims = None (float or array-like): the dimensions of the 
            box. If this is None, the current box volume is used along all
            dimensions. If it is a float, this is taken as the box length
            along all dimensions. If it is an array-like, the elements are
            taken as the box length along each axis.
        * los_axis = 0 (integer): the line-of-sight axis
        
    Returns:
        A tuple with (P0, P2, P4, k) where P0, P2 and P4
        are the multipole moments as a function of k and
        k is midpoints of the k bins
        All four arrays have the same length
    '''
    
    #First calculate the power spectrum
    box_dims = _get_dims(box_dims, input_array.shape)
    ps = power_spectrum_nd(input_array, box_dims)
    
    #Get k values and bins
    k_comp, k = _get_k(input_array, box_dims)
    kbins = _get_kbins(kbins, box_dims, k)
    dk = (kbins[1:]-kbins[:-1])/2.
    mu = _get_mu(k_comp, k, los_axis)
    
    print k.min(), k.max(), kbins
    
    #Legendre polynomials
    P0 = np.ones_like(mu)
    P2 = 0.5*(3.*mu**2 - 1.)
    P4 = 4.375*(mu**2-0.115587)*(mu**2-0.741556) 
    
    #Bin data
    n_kbins = len(kbins)-1
    outdata_P0 = np.zeros(n_kbins)
    outdata_P2 = np.zeros_like(outdata_P0)
    outdata_P4 = np.zeros_like(outdata_P0) 
    
    for i in range(n_kbins):
        idx = (k > kbins[i]) * (k <= kbins[i+1])
        outdata_P0[i] = np.sum(ps[idx]*P0[idx])/np.sum(P0[idx]**2)
        outdata_P2[i] = np.sum(ps[idx]*P2[idx])/np.sum(P2[idx]**2)
        outdata_P4[i] = np.sum(ps[idx]*P4[idx])/np.sum(P4[idx]**2)
        
    #Normalize
    
    return outdata_P0, outdata_P2, outdata_P4, kbins[:-1]+dk



#_________TEST_______________

if __name__ == '__main__':
    import c2raytools as c2t
    import pylab as pl
    
    #Load sumans data
    z = 9.026
    suman_dir = '/disk/dawn-1/smaju/halo_test/114Mpc/'
    suman_dT = c2t.read_binary_with_meshinfo(suman_dir + \
                                             'dT_files/sem_num_e0.0/%.3f_dT_rsd.cbin' % z, order='c')
    suman_dT -= suman_dT.mean()
    suman_ps = np.loadtxt(suman_dir + 'power_mom_out/sem_num_e0.0/pk_mom_%.3f.txt' % z)
    suman_k = suman_ps[:,0]
    suman_P0 = suman_ps[:,1]
    suman_P2 = suman_ps[:,2]
    suman_P4 = suman_ps[:,3]

    #Calculate    
    c2t.set_sim_constants(114.)
    k_min = 2*np.pi/c2t.LB
    k_max = np.pi/(c2t.LB/256.)
    kdelta = np.log10(k_max/k_min)/15
    bin_edges = k_min * 10**(np.arange(16)*kdelta)
    P0, P2, P4, k = power_spectrum_multipoles(suman_dT, kbins=bin_edges, los_axis=2)
    print k
    
#    #Test
#    x_HI = 0.325
#    P0 /= x_HI**2
#    P2 /= x_HI**2
#    P4 /= x_HI**2
    
#    Plot
#    pl.semilogx(suman_k, suman_P0, 'g-')
#    pl.semilogx(suman_k, suman_P2, 'b-')
#    pl.semilogx(suman_k, suman_P4, 'r-')
#    pl.semilogx(suman_k, suman_P2/suman_P0, 'b-')
#    
#    pl.semilogx(k, P0, 'g:')
#    pl.semilogx(k, P2, 'b:')
#    pl.semilogx(k, P4, 'r:')
#    pl.semilogx(k, P2/P0, 'b:')
#    
    pl.semilogx(k, P0/suman_P0)
    pl.semilogx(k, P2/suman_P2)
    pl.semilogx(k, P4/suman_P4)
    
    pl.xlim([0.4, 10])
    pl.ylim([0., 2.2])
#    pl.ylim([-1e4, 6e4])
    pl.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        