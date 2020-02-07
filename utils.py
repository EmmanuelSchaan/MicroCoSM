from headers import *

##################################################################################
#  Mathematical functions



def W3d_sth(x):
   """Fourier transform of a 3d spherical top hat window function.
   Use x = k*R as input,
   where R is the tophat radius and k the wave vector.
   Input and output are dimensionless.
   """
   if x < 1.e-3:  # for small x, replace by expansion, for numerical stability
      f = 1. - 0.1* x**2 + 0.00357143* x**4
   else:
      f = (3./(x**3)) * ( np.sin(x) - x * np.cos(x) )
   return f


def dW3d_sth(x):
   """Derivative of the FT of the top hat.
   Input and output are dimensionless.
   """
   f = 3. * (3. * x * np.cos(x) - 3. * np.sin(x) + (x**2) * np.sin(x)) / (x**4)
   return f


def W2d_cth(x):
   """FT of a 2d circular top hat window function.
   Input and output are dimensionless.
   """
   return 2.*special.jn(1, x) / x

def W1d_th(x):
   """FT of a 1d tophat
   normalized to unity at k=0 (ie real space integral is 1)
   Input and output are dimensionless.
   """
   return sinc(x/2.)
   
def Si(x):
   return special.sici(x)[0]

def Ci(x):
   return special.sici(x)[1]

def sinc(x):
   return special.sph_jn(0, x)[0][0]

def j0(x):
   """relevant for isotropic Fourier transform in 2d
   """
   return special.jn(0, x)


def i0(x):
   """Modified Bessel function of the first kind
   """
   return special.iv(0, x)


##################################################################################
# formatting numbers

def intExpForm(input):
   """
   clean scientific notation for file names
   removes trailing decimal point if not needed
   """
   a = '%e' % np.float(input)
   # mantissa: remove trailing zeros
   # then remove dot if no decimal digits
   mantissa = a.split('e')[0].rstrip('0').rstrip('.')
   # exponent: remove + sign if there, and leading zeros
   exponent = np.int(a.split('e')[1])
   exponent = np.str(exponent)
   if exponent=='0':
      return mantissa
   else:
      return mantissa + 'e' + exponent



def floatExpForm(input):
   """same as intExpForm, except always leaves the decimal point
   """
   a = '%e' % np.float(input)
   # mantissa: remove trailing zeros
   # then remove dot if no decimal digits
   mantissa = a.split('e')[0].rstrip('0')
   # exponent: remove + sign if there, and leading zeros
   exponent = np.int(a.split('e')[1])
   exponent = np.str(exponent)
   if exponent=='0':
      return mantissa
   else:
      return mantissa + 'e' + exponent


##################################################################################
# Matrix inversion for ill-conditioned matrices, with SVD

def invertMatrixSvdTruncated(matrix, epsilon=1.e-5, keepLow=True):
   '''Invert a matrix by inverting its SVD,
   and setting to zero the singular values that are too small/large.
   epsilon sets the tolerance for discarding singular values.
   keepLow=True: for inverting a cov matrix, want to keep the modes with lowest variance.
   keepLow=False: for inverting a Fisher maytrix, want to keep the modes with highest Fisher information.
   '''
   # Perform SVD on matrix
   U, s, Vh = scipy.linalg.svd(matrix)
   # invert the singular values
   sInv = 1./s
   # remove the super poorly constrained modes, that lead to numerical instabilities
   if keepLow:
      sInvMax = np.max(sInv)
      sInv[sInv<=sInvMax*epsilon] = 0.
   else:
      sInvMin = np.min(sInv)
      sInv[sInv>=sInvMin/epsilon] = 0.
   # create diagonal matrix
   sInv = np.diag(sInv)
   # invert the hermitian matrices
   V = np.conj(Vh.transpose())
   Uh = np.conj(U.transpose())
   # generate the inverse
   result = np.dot(V, np.dot(sInv, Uh))
   return result


##################################################################################
# Generating ell bins with constant number of modes


def generateEllBins(lMin, lMax, nL, fsky=1.):
   '''Generates nL bins between lMin and lMax,
   such that the number of 2d modes in each bin is identical.
   Returns the bin centers, the bin edges, the bin widths, and the number of modes per bin.
   '''
   # area in ell space between lMin and l,
   # normalized to 1 when l=lMax
   farea = lambda l: (l**2 - lMin**2) / (lMax**2 - lMin**2)

   # find the bin edges,
   # such that each bin has equal number of modes
   Le = np.zeros(nL+1)
   for iL in range(nL+1):
      f = lambda l: farea(l) - float(iL) / nL
      Le[iL] = optimize.brentq(f , lMin, lMax)
   
   # use the average ell in the bin, weighted by number of modes, as bin center
   Lc =  2./3. * (Le[1:]**3 - Le[:-1]**3) / (Le[1:]**2 - Le[:-1]**2)
   # bin spacing
   dL = Le[1:] - Le[:-1]
   # Number of modes
   lF = 2.*np.pi / np.sqrt(4. * np.pi * fsky)
   nModesTotal = np.pi * (lMax**2 - lMin**2) / lF**2
   Nmodes = nModesTotal / nL * np.ones(nL)   
   
   return Lc, dL, Nmodes, Le


##################################################################################
#  Extract non-mask data vector and cov matrix

def extractMaskedMat(cov, mask=None, I=None):
   '''cov: large matrix
   mask: 1d array, 0 if unmasked, anything else if masked
   I: indices of the large matrix to keep, pre-masking
   '''
   # convert mask to 0 and 1
   mask = mask.astype(bool)
   # extract indices of interest, if needed
   if I is not None:
      mask = mask[I]
      J = np.ix_(I, I)
      cov = cov[J]
   if mask is not None:
      # nb of unmasked rows
      nNew = np.int(np.sum(1 - mask))
      # mask cov matrix
      mask = np.diag(mask)
      cov = ma.masked_array(cov, mask=mask)
      cov = ma.mask_rowcols(cov)
      # extract the non-masked elements
      cov = cov.compressed().reshape((nNew, nNew))
   return cov

def extractMaskedVec(vec, mask=None, I=None):
   '''vec: large vector
   mask: 1d array, 0 if unmasked, anything else if masked
   I: indices of the large vector to keep, pre-masking
   '''
   # convert mask to 0 and 1
   mask = mask.astype(bool)
   # extract indices of interest, if needed
   if I is not None:
      mask = mask[I]
      vec = vec[I]
   if mask is not None:
      # mask vec matrix
      vec = ma.masked_array(vec, mask=mask)
      # extract the non-masked elements
      vec = vec.compressed()
   return vec

##################################################################################
# Measuring RAM usage of the current process

def currentRssMB(pid=None):
   """Returns the RSS (resident set size, ie portion of RAM occupied by a process).
   Output in MB.
   """
   memory = dict(psutil.Process(pid=None).memory_info()._asdict())
   result = memory['rss']  # in Bytes
   return result / 1.e6


##################################################################################

def divide(x, y, exceptOut=0.):
   '''Returns 0. or any requested value
   when dividing by zero.
   '''
   try: return x/y
   except ZeroDivisionError: return exceptOut

