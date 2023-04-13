#from pylab import *


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import scipy
from scipy import special, optimize, integrate, stats
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, interp1d, interp2d, BarycentricInterpolator
from time import time
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import matplotlib.colors as mc
import colorsys
from timeit import timeit
from time import time
from copy import copy
import sys

#The bivariate_normal is no longer available in mlab.py
#from matplotlib.mlab import bivariate_normal 
#We can simply create it
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


##################################################################################
# for pretty plots

#from matplotlib import rc
##rc('font',**{'size':'20','family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
#rc('font',**{'size':'22','family':'serif','serif':['CMU serif']})
#rc('mathtext', **{'fontset':'cm'})
#rc('text', usetex=True)
##rc('text.latex', preamble='/usepackage{amsmath}, /usepackage{amssymb}')
##rc('font', size=20)
#rc('legend',**{'fontsize':'18'})
#
## fonty stuffs
##font.serif: CMU Serif
##font.family: serif
##mathtext.fontset: cm
##text.usetex: False
##text.latex.preamble: /usepackage{amsmath}

def darkerLighter(color, amount=0.):
   """
   Adapted from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
   Input can be matplotlib color string, hex string, or RGB tuple.
   amount=0: color unchanged
   amount=-1: returns white
   amount=1: returns black

   Examples:
   >> lighten_color('g', 0.3)
   >> lighten_color('#F034A3', 0.6)
   >> lighten_color((.3,.55,.1), 0.5)
   """
   # force amount between 0. and 1.
   amount = min(amount, 1.)
   amount = max(amount, -1.)
   # read the color
   try:
      c = mc.cnames[color]
   except:
      c = color
   c = colorsys.rgb_to_hls(*mc.to_rgb(c))
   # my Lagrange interpolation polynomial
   newC1 = 0.5*amount*(amount-1.) - c[1]*(amount+1.)*(amount-1.)
   return colorsys.hls_to_rgb(c[0], newC1, c[2])


##################################################################################

from importlib import reload

import utils
reload(utils)
from utils import *

