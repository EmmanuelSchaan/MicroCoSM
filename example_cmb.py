from importlib import reload
import cmb
reload(cmb)
from cmb import *


##################################################################################
# CMB specifications

# ACT 148GHz
#cmb = CMB(beam=1.4, noise=12., nu1=148.e9, nu2=148.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, fg=False, atm=False, name="act148")

# Planck SMICA map
#cmb = CMB(beam=5., noise=60., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, fg=False, atm=False, name="plancksmica")

# ACTPol
#cmb = CMB(beam=1.4, noise=18., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, fg=False, atm=False, name="actpol")

# AdvACT
#cmb = CMB(beam=1.4, noise=10., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, fg=False, atm=False, name="actpol")

# CMB S4
cmb = CMB(beam=1., noise=1., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, fg=True, atm=False, name="cmbs4")

# the "reference CMB experiment" from Hu Okamoto 2002
#cmb = CMB(beam=4., noise=1., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, fg=False, atm=False, name="cmbhuokamoto02")


##################################################################################
# Frequency dependence of the various components

cmb.plotFreqDpdce()


##################################################################################
# Power spectra

# TT, EE, TE and BB
cmb.plotCl()

# TT: various foreground components
cmb.plotClTT()
