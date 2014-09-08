
import numpy as np
from power_spectrum import _get_dims, power_spectrum_nd, _get_k,\
 _get_kbins, _get_mu, get_eval

def power_spectrum_multipoles(input_array, kbins = 10, box_dims = None,\
                               los_axis = 0):
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
        k contains the midpoints of the k bins.
        All four arrays have the same length
    '''
    
    assert(los_axis >= 0 and los_axis <= len(input_array.shape))
    
    #First calculate the power spectrum
    box_dims = _get_dims(box_dims, input_array.shape)
    ps = power_spectrum_nd(input_array, box_dims)
    
    #Get k values and bins
    k_comp, k = _get_k(input_array, box_dims)
    kbins = _get_kbins(kbins, box_dims, k)
    dk = (kbins[1:]-kbins[:-1])/2.
    mu = _get_mu(k_comp, k, los_axis)
    
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
        kmin = kbins[i]
        kmax = kbins[i+1]
        idx = get_eval()('(k >= kmin) & (k < kmax)')
        outdata_P0[i] = np.sum(ps[idx]*P0[idx])/np.sum(P0[idx]**2)
        outdata_P2[i] = np.sum(ps[idx]*P2[idx])/np.sum(P2[idx]**2)
        outdata_P4[i] = np.sum(ps[idx]*P4[idx])/np.sum(P4[idx]**2)
        
    
    return outdata_P0, outdata_P2, outdata_P4, kbins[:-1]+dk


    
    
    
    
    
    
    
    
    
        