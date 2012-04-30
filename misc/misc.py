import numpy as np
def gauss_kern(size, sizey = None, sigma=1.0):
	''' Return a normalized gaussian kernel. 
	if sizey is not set, it will be the same as sizex '''

	size = int(size/2)
	if not sizey:
		sizey = size
	else:
		sizey = int(sizey/2)

	x,y = np.mgrid[-size:size, -sizey:sizey]
	#g = exp(-(x**2/float(size) + y**2/float(sizey))/sigma**2)
	g = np.exp(-(x**2 + y**2)/sigma**2)

	return g/g.sum()
