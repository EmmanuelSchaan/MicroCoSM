from headers import *

###############################################################################
# class containing all templates for CMB components at various frequencies
# all the maps are debeamed, ie a shot noise component is ell-independent,
# and the detector noise grows exponentially with the beam.
# Signal and noise are assumed to be in muK^2*steradian.

class CMB(object):
   
   def __str__(self):
      return self.name
   
   def __init__(self, beam=1., noise=1., nu1=143.e9, nu2=143.e9, lMin=30., lMaxT=3.e3, lMaxP=5.e3, fg=True, atm=False, name=None):
   
      # name
      self.name = "cmb_beam"+str(round(beam, 3))+"_noise"+str(round(noise, 3))+ "_nu"+str(np.int(nu1/1.e9))+"_nu"+str(np.int(nu2/1.e9))+"_lmin"+str(np.int(lMin))+"_lmaxT"+str(int(lMaxT))+"_lmaxP"+str(int(lMaxP))
      if atm:
         self.name += "_atmnoise"
      if name is not None:
         self.name += "_"+name
      # frequencies in Hz
      self.nu1 = nu1
      self.nu2 = nu2
      # beam fwhm in radians (1 arcmin)
      self.fwhm = beam * (np.pi/180.)/60.
      # detector sensitivity in muK*rad.
      self.sensitivity = noise * (np.pi/180.)/60.
      # ell limits
      self.lMin = lMin
      self.lMaxT = lMaxT
      self.lMaxP = lMaxP
      
      ##################################################################################
      
      # Physical constants
      self.c = 3.e8  # m/s
      self.h = 6.63e-34 # SI
      self.kB = 1.38e-23   # SI
      self.Tcmb = 2.726 # K
#!!! manuwarning
      self.Jansky = 1.e-26 # W/m^2/Hz
      self.Jy = 1.e-26  # [W/m^2/Hz]
      
      # Ref frequency in the Dunkley+13 foreground model
      self.nu0 = 150.e9
      
      # convert from Dl to Cl: Dl = l(l+1) Cl / 2pi
      self.fdl_to_cl = lambda l: 2. * np.pi / (l*(l+1.))
   
      ##################################################################################
      ##################################################################################
      # interpolate the frequency dependencies, for speed
      
      self.pathFreqDpdces = "./input/cmb/freq_dpdces.txt"
      if not os.path.exists(self.pathFreqDpdces):
         self.saveFreqDpdce()
      self.loadFreqDpdce()


      ##################################################################################
      ##################################################################################
      # unlensed primary T, E, B

      # read the Dl, and convert to Cl
      data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
      factor = self.fdl_to_cl(data[:,0])
      data[:,1] *= factor
      data[:,2] *= factor
      data[:,3] *= factor
      data[:,4] *= factor

      # interpolate
      self.funlensedTT = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      self.funlensedEE = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=False, fill_value=0.)
      self.funlensedBB = interp1d(data[:,0], data[:,3], kind='linear', bounds_error=False, fill_value=0.)
      self.funlensedTE = interp1d(data[:,0], data[:,4], kind='linear', bounds_error=False, fill_value=0.)

      
      ##################################################################################
      # lensed primary T, E, B

      # read the Dl, and convert to Cl
      data = np.genfromtxt("./input/universe_Planck15/camb/lensedCls.dat")
      factor = self.fdl_to_cl(data[:,0])
      data[:,1] *= factor
      data[:,2] *= factor
      data[:,3] *= factor
      data[:,4] *= factor

      # interpolate
      self.flensedTT = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      self.flensedEE = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=False, fill_value=0.)
      self.flensedBB = interp1d(data[:,0], data[:,3], kind='linear', bounds_error=False, fill_value=0.)
      self.flensedTE = interp1d(data[:,0], data[:,4], kind='linear', bounds_error=False, fill_value=0.)

      ##################################################################################
      # total primary T, E, B: lensed + noise + atm + fg

      # TT
      if fg and not atm:
         self.ftotalTT = lambda l: self.flensedTT(l) + self.fkSZ(l) + self.fCIB(l) + self.ftSZ(l) + self.ftSZ_CIB(l) + self.fradioPoisson(l) + self.fdetectorNoise(l)
      elif fg and atm:
         self.ftotalTT = lambda l: self.flensedTT(l) + self.fkSZ(l) + self.fCIB(l) + self.ftSZ(l) + self.ftSZ_CIB(l) + self.fradioPoisson(l) + self.fdetectorNoise(l) + self.fatmosphericNoiseTT(l)
      elif not fg and not atm:
         self.ftotalTT = lambda l: self.flensedTT(l) + self.fdetectorNoise(l)
      elif not fg and atm:
         self.ftotalTT = lambda l: self.flensedTT(l)+ self.fdetectorNoise(l) + self.fatmosphericNoiseTT(l)
      
      # TE, EE, BB
      if atm:
         self.ftotalEE = lambda l: self.flensedEE(l) + 2.*self.fdetectorNoise(l) + self.fatmosphericNoisePP(l)
         self.ftotalBB = lambda l: self.flensedBB(l) + 2.*self.fdetectorNoise(l) + self.fatmosphericNoisePP(l)
         self.ftotalTE = lambda l: self.flensedTE(l)
      else:
         self.ftotalEE = lambda l: self.flensedEE(l) + 2.*self.fdetectorNoise(l)
         self.ftotalBB = lambda l: self.flensedBB(l) + 2.*self.fdetectorNoise(l)
         self.ftotalTE = lambda l: self.flensedTE(l)
      

      ###############################################################################
      ###############################################################################
      # Foreground power spectra

      # kSZ: Dunkley et al 2013
      data = np.genfromtxt("./input/cmb/digitizing_SZ_template/kSZ.txt")   # read l, Dl
      data[:,1] *= self.fdl_to_cl(data[:,0]) # convert Dl to Cl
      fkSZ_template = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
      a_kSZ = 1.5  # 1.5 predicted by Battaglia et al 2010. Upper limit from Dunkley+13 is 5.
      self.fkSZ = lambda l: a_kSZ * fkSZ_template(l)
      

      # tSZ: Dunkley et al 2013
      data = np.genfromtxt("./input/cmb/digitizing_SZ_template/tSZ.txt")   # read l, Dl
      data[:,1] *= self.fdl_to_cl(data[:,0]) # convert Dl to Cl
      ftSZ_template = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
      a_tSZ = 4.0
      self.ftSZ = lambda l: a_tSZ * ftSZ_template(l) * self.tszFreqDpdceT(self.nu1) * self.tszFreqDpdceT(self.nu2) / self.tszFreqDpdceT(self.nu0)**2


      # tSZ x CIB: Dunkley et al 2013
      xi = 0.2 # upper limit at 95% confidence
      a_tSZ = 4.0
      a_CIBC = 5.7
      betaC = 2.1
      Td = 9.7
      # watch for the minus sign
      data = np.genfromtxt ("./input/cmb/digitizing_tSZCIB_template/minus_tSZ_CIB.txt")   # read l, Dl
      data[:,1] *= self.fdl_to_cl(data[:,0]) # convert Dl to Cl
      ftSZCIB_template = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
      
      self.ftSZ_CIB = lambda l: -2. * xi * np.sqrt(a_tSZ * a_CIBC) * ftSZCIB_template(l) * (self.tszFreqDpdceT(self.nu1)*self.cibPoissonFreqDpdceT(self.nu2) + self.tszFreqDpdceT(self.nu2)*self.cibPoissonFreqDpdceT(self.nu1)) / (2.* self.tszFreqDpdceT(self.nu0)*self.cibPoissonFreqDpdceT(self.nu0))
         

   ###############################################################################

   def fCIBPoisson(self, l):
      '''CIB Poisson
      '''
      a_CIBP = 7.0
      Td = 9.7
      betaP = 2.1
      result = a_CIBP * (l/3000.)**2 * self.fdl_to_cl(l)
      result *= self.cibPoissonFreqDpdceT(self.nu1) * self.cibPoissonFreqDpdceT(self.nu2) / self.cibPoissonFreqDpdceT(self.nu0)**2
      return result
   

   def fCIBClustered(self, l, nu1=None, nu2=None):
      '''CIB Clustered
      '''
      a_CIBC = 5.7
      n = 1.2
      Td = 9.7
      betaC = 2.1
      result = a_CIBC * (l/3000.)**(2-n) * self.fdl_to_cl(l)
      result *= self.cibClusteredFreqDpdceT(self.nu1) * self.cibClusteredFreqDpdceT(self.nu2) / self.cibClusteredFreqDpdceT(self.nu0)**2
      return result


   def fCIB(self, l):
      '''CIB Poisson + Clustered
      '''
      return self.fCIBPoisson(l) + self.fCIBClustered(l)


   def fradioPoisson(self, l):
      '''Radio Poisson
      '''
      alpha_s = -0.5
      a_s = 3.2
      result = a_s * (l/3000.)**2* self.fdl_to_cl(l)
      result *= self.radioPoissonFreqDpdceT(self.nu1) * self.radioPoissonFreqDpdceT(self.nu2) / self.radioPoissonFreqDpdceT(self.nu0)**2
      return result


   def fgalacticDust(self, l):
      '''Galactic dust
      '''
      beta_g = 3.8
      n_g = -0.7
      a_ge = 0.9
      a_gs = 0.7  # 95% confidence limit
      result = a_gs * (l/3000.)**2 * self.fdl_to_cl(l)
      result *= self.galacticDustFreqDpdceT(self.nu1) * self.galacticDustFreqDpdceT(self.nu2) / self.galacticDustFreqDpdceT(self.nu0)**2
      return result



   
   ###############################################################################
   ###############################################################################
   # atmospheric noise in temperature and polarization
   # only implemented for 150GHz
   # from Matthew Hasselfield's model for Simons observatory
   # getAtmosphere function from Mathew Madhavacheril
   
   def getAtmosphere(self):
      '''Get TT-lknee, TT-alpha, PP-lknee, PP-alpha
      '''
      # best fits from M.Hasselfield
      size = np.array([0.5,5.,7.]) # telescope size in meters
      ttalpha = -4.7
      ppalpha = np.array([-2.6,-3.8,-3.9])
      ttlknee = np.array([350.,3400.,4900.])
      pplknee = np.array([60,330,460])

      # convert telescope size to beam
      cspeed = 299792458.  # m/s
      wavelength = cspeed/self.nu1  # m
      resin = 1.22*wavelength/size  # beam fwhm in rad
      
      # interpolate Matt's fits
      ttlkneeFunc = interp1d(resin,ttlknee,fill_value="extrapolate",kind="linear")
      ttalphaFunc = lambda x: ttalpha
      pplkneeFunc = interp1d(resin,pplknee,fill_value="extrapolate",kind="linear")
      ppalphaFunc = interp1d(resin,ppalpha,fill_value="extrapolate",kind="linear")
      
      b = self.fwhm  # beam fwhm in rad
      return ttlkneeFunc(b),ttalphaFunc(b),pplkneeFunc(b),ppalphaFunc(b)
   
   
   def fatmosphericNoiseTT(self, l):
      lKnee, alpha, _, _ = self.getAtmosphere()
      result = (lKnee/l)**(-alpha)
      result *= self.fdetectorNoise(l)
      return result

   def fatmosphericNoisePP(self, l):
      _, _, lKnee, alpha = self.getAtmosphere()
      result = (lKnee/l)**(-alpha)
      result *= self.fdetectorNoise(l)
      result *= 2.   # noise is larger in polarization
      return result
   

   ###############################################################################
   ###############################################################################
   # beam and detector noise

   def fbeamTheta(self, theta, fwhm=None):
      if fwhm is None:
         fwhm = self.fwhm
      sigma_beam = fwhm / np.sqrt(8.*np.log(2.))
      return np.exp(-0.5*theta**2/sigma_beam**2) / (2.*np.pi*sigma_beam**2)
   
   def fbeam(self, l, fwhm=None):
      if fwhm==0.:
         return 1.
      elif fwhm is None:
         fwhm = self.fwhm
      sigma_beam = self.fwhm / np.sqrt(8.*np.log(2.))
      return np.exp(-0.5*l**2 * sigma_beam**2)
   
   def fdetectorNoise(self, l):
      return self.sensitivity**2 / self.fbeam(l)**2


   ##################################################################################
   ##################################################################################
   # Black body intensity and conversions


   def blackbody(self, nu, T):
      '''blackbody function
      input: nu [Hz], T thermo temperature of the black body [K]
      output in SI: [W / Hz / m^2 / sr]
      '''
      x = self.h*nu/(self.kB*T)
      result = 2.*self.h*nu**3 /self.c**2
      result /= np.exp(x) - 1.
      return result

   def dBdT(self, nu, T):
      '''d(blackbody)/dT, such that
      dI = d(blackbody)/dT * dT
      input: nu [Hz], T thermo temperature of the black body [K]
      output in SI: [W / Hz / m^2 / sr / K]
      '''
      x = self.h*nu/(self.kB*T)
      result = 2.*self.h**2*nu**4
      result /= self.kB*T**2*self.c**2
      result *= np.exp(x) / (np.exp(x) - 1.)**2
      return result

   def dlnBdlnT(self, nu, T):
      '''dlnBlackbody/dlnT
      input: nu [Hz], T thermo temperature of the black body [K]
      output is dimensionless
      '''
      x = self.h*nu/(self.kB*T)
      return x * np.exp(x) / (np.exp(x) - 1.)

   def dBdTrj(self, nu, T):
      '''d(blackbody)/dTrj, where Trj is the Rayleigh-Jeans brightness temperature,
       such that:
      dI = d(blackbody)/dTrj * dTrj
      input: nu [Hz], T thermo temperature [K]
      output in SI: [W / Hz / m^2 / sr / Krj]
      '''
      result = 2. * nu**2 * self.kB / self.c**2
      return result


   ##################################################################################
   # Conversion between SI, Jy/sr, Kthermo, Krj

   def convertIntSITo(self, nu, kind="intSI"):
      '''kind: "intSI", "intJy/sr", "tempKcmb", "tempKrj"
      '''
      if kind=="intSI":
         result = 1.
      elif kind=="intJy/sr":
         result = 1. / self.Jy
      elif kind=="tempKcmb":
         result = 1. / self.dBdT(nu, self.Tcmb)
      elif kind=="tempKrj":
         result = 1. / self.dBdTrj(nu, self.Tcmb)
      return result


   ##################################################################################
   # Frequency dependences of the various components, in intensity

   def cmbFreqDpdceInt(self, nu):
      '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
      arbitrary normalization
      '''
      return self.blackbody(nu, self.Tcmb)

   def kszFreqDpdceInt(self, nu):
      '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
      arbitrary normalization
      '''
      return self.blackbody(nu, self.Tcmb)

   def tszFreqDpdceInt(self, nu):
      '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
      arbitrary normalization
      '''
      # freq dpdce such that dT/T = (freq dpdce) * 2.*y
      x = self.h*nu/(self.kB*self.Tcmb)
      result = x*(np.exp(x)+1.)/(np.exp(x)-1.) -4.
      # freq dpdce such that dT = (freq dpdce) * 2.*y
      result *= self.Tcmb
      # freq dpdce such that dI = (freq dpdce) * 2.*y
      result *= self.blackbody(nu, self.Tcmb)
      return result

   def cibPoissonFreqDpdceInt(self, nu):
      '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
      arbitrary normalization
      '''
      Td = 9.7
      betaP = 2.1
      return nu**betaP * self.blackbody(nu, Td)
   

   def cibClusteredFreqDpdceInt(self, nu):
      '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
      arbitrary normalization
      '''
      Td = 9.7
      betaC = 2.1
      return nu**betaC * self.blackbody(nu, Td)

   def radioPoissonFreqDpdceInt(self, nu):
      '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
      arbitrary normalization
      '''
      alpha_s = -0.5
      return nu**alpha_s

   def galacticDustFreqDpdceInt(self, nu):
      '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
      arbitrary normalization
      '''
      beta_g = 3.8
      return nu**beta_g


   ##################################################################################


   def saveFreqDpdce(self):
      '''Precompute the various frequency dependences, in the various units.
      '''
      # Frequencies to evaluate
      self.nNu = 501
      self.Nu = np.logspace(np.log10(0.1), np.log10(1.e4), self.nNu, 10.)*1.e9   # in Hz

      data = np.zeros((self.nNu, 22))
      data[:,0] = self.Nu.copy()

      # Intensity freq dpdces
      data[:,1] = np.array(map(self.cmbFreqDpdceInt, self.Nu))
      data[:,2] = np.array(map(self.kszFreqDpdceInt, self.Nu))
      data[:,3] = np.array(map(self.tszFreqDpdceInt, self.Nu))
      data[:,4] = np.array(map(self.cibPoissonFreqDpdceInt, self.Nu))
      data[:,5] = np.array(map(self.cibClusteredFreqDpdceInt, self.Nu))
      data[:,6] = np.array(map(self.radioPoissonFreqDpdceInt, self.Nu))
      data[:,7] = np.array(map(self.galacticDustFreqDpdceInt, self.Nu))

      # Thermo temperature freq dpdces
      f = lambda nu: self.convertIntSITo(nu, kind="tempKcmb")
      intToTemp = np.array(map(f, self.Nu))
      #
      data[:,8] = data[:,1] * intToTemp
      data[:,9] = data[:,2] * intToTemp
      data[:,10] = data[:,3] * intToTemp
      data[:,11] = data[:,4] * intToTemp
      data[:,12] = data[:,5] * intToTemp
      data[:,13] = data[:,6] * intToTemp
      data[:,14] = data[:,7] * intToTemp

      # Rayleigh-Jeans temperature freq dpdces
      f = lambda nu: self.convertIntSITo(nu, kind="tempKrj")
      intToTempRJ = np.array(map(f, self.Nu))
      #
      data[:,15] = data[:,1] * intToTempRJ
      data[:,16] = data[:,2] * intToTempRJ
      data[:,17] = data[:,3] * intToTempRJ
      data[:,18] = data[:,4] * intToTempRJ
      data[:,19] = data[:,5] * intToTempRJ
      data[:,20] = data[:,6] * intToTempRJ
      data[:,21] = data[:,7] * intToTempRJ

      
      np.savetxt(self.pathFreqDpdces, data)


   def loadFreqDpdce(self):
      data = np.genfromtxt(self.pathFreqDpdces)
      self.Nu = data[:,0]

      # interpolate the freq dpdces for intensity
      self.cmbFreqDpdceI = interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False, fill_value=0.)
      self.kszFreqDpdceI = interp1d(data[:,0], data[:,2], kind='linear', bounds_error=False, fill_value=0.)
      self.tszFreqDpdceI = interp1d(data[:,0], data[:,3], kind='linear', bounds_error=False, fill_value=0.)
      self.cibPoissonFreqDpdceI = interp1d(data[:,0], data[:,4], kind='linear', bounds_error=False, fill_value=0.)
      self.cibClusteredFreqDpdceI = interp1d(data[:,0], data[:,5], kind='linear', bounds_error=False, fill_value=0.)
      self.radioPoissonFreqDpdceI = interp1d(data[:,0], data[:,6], kind='linear', bounds_error=False, fill_value=0.)
      self.galacticDustFreqDpdceI = interp1d(data[:,0], data[:,7], kind='linear', bounds_error=False, fill_value=0.)

      # interpolate the freq dpdces for thermodynamical temperature
      self.cmbFreqDpdceT = interp1d(data[:,0], data[:,8], kind='linear', bounds_error=False, fill_value=0.)
      self.kszFreqDpdceT = interp1d(data[:,0], data[:,9], kind='linear', bounds_error=False, fill_value=0.)
      self.tszFreqDpdceT = interp1d(data[:,0], data[:,10], kind='linear', bounds_error=False, fill_value=0.)
      self.cibPoissonFreqDpdceT = interp1d(data[:,0], data[:,11], kind='linear', bounds_error=False, fill_value=0.)
      self.cibClusteredFreqDpdceT = interp1d(data[:,0], data[:,12], kind='linear', bounds_error=False, fill_value=0.)
      self.radioPoissonFreqDpdceT = interp1d(data[:,0], data[:,13], kind='linear', bounds_error=False, fill_value=0.)
      self.galacticDustFreqDpdceT = interp1d(data[:,0], data[:,14], kind='linear', bounds_error=False, fill_value=0.)

      # interpolate the freq dpdces for Rayleigh-Jeans temperature
      self.cmbFreqDpdceTrj = interp1d(data[:,0], data[:,15], kind='linear', bounds_error=False, fill_value=0.)
      self.kszFreqDpdceTrj = interp1d(data[:,0], data[:,16], kind='linear', bounds_error=False, fill_value=0.)
      self.tszFreqDpdceTrj = interp1d(data[:,0], data[:,17], kind='linear', bounds_error=False, fill_value=0.)
      self.cibPoissonFreqDpdceTrj = interp1d(data[:,0], data[:,18], kind='linear', bounds_error=False, fill_value=0.)
      self.cibClusteredFreqDpdceTrj = interp1d(data[:,0], data[:,19], kind='linear', bounds_error=False, fill_value=0.)
      self.radioPoissonFreqDpdceTrj = interp1d(data[:,0], data[:,20], kind='linear', bounds_error=False, fill_value=0.)
      self.galacticDustFreqDpdceTrj = interp1d(data[:,0], data[:,21], kind='linear', bounds_error=False, fill_value=0.)


   def plotFreqDpdce(self, lMin=2.e3, lMax=3.e3):
      '''Computes the RMS temperature fluctuation of each component
      between the specified lMin and lMax,
      then shows its frequency dependence.
      '''
      
      # RMS temperature fluctuations in muK, at current experiment frequencies,
      # with current experimental beam
      # within the lMax and lMin chosen
      dTcmb = self.rmsT(self.flensedTT, lMin=lMin, lMax=lMax, fwhm=None)
      dTksz = self.rmsT(self.fkSZ, lMin=lMin, lMax=lMax, fwhm=None)
      dTtsz = self.rmsT(self.ftSZ, lMin=lMin, lMax=lMax, fwhm=None)
      dTcib = self.rmsT(self.fCIB, lMin=lMin, lMax=lMax, fwhm=None)
      dTradiops = self.rmsT(self.fradioPoisson, lMin=lMin, lMax=lMax, fwhm=None)
      
      
      # Temperatures [muKcmb], as a function of freq
      DTcmb = dTcmb * self.cmbFreqDpdceT(self.Nu)
      DTcmb /= np.sqrt( self.cmbFreqDpdceT(self.nu1) * self.cmbFreqDpdceT(self.nu2) )
      #
      DTksz = dTksz * self.kszFreqDpdceT(self.Nu)
      DTksz /= np.sqrt( self.kszFreqDpdceT(self.nu1) * self.kszFreqDpdceT(self.nu2) )
      #
      DTtsz = dTtsz * self.tszFreqDpdceT(self.Nu)
      DTtsz /= np.sqrt( self.tszFreqDpdceT(self.nu1) * self.tszFreqDpdceT(self.nu2) )
      #
      DTcib = dTcib * self.cibPoissonFreqDpdceT(self.Nu)
      DTcib /= np.sqrt( self.cibPoissonFreqDpdceT(self.nu1) * self.cibPoissonFreqDpdceT(self.nu2) )
      #
      DTradiops = dTradiops * self.radioPoissonFreqDpdceT(self.Nu)
      DTradiops /= np.sqrt( self.radioPoissonFreqDpdceT(self.nu1) * self.radioPoissonFreqDpdceT(self.nu2) )
      
      
      # Intensities [Jy/sr], as a function of freqs
      # convert from muK to Jy/sr
      factor = 1.e-6 # convert to K
      factor /= self.convertIntSITo(self.Nu, kind="tempKcmb")  # convert to SI
      factor *= self.convertIntSITo(self.Nu, kind="intJy/sr")  # convert to Jy/sr
      #
      DIcmb = DTcmb * factor
      DIksz = DTksz * factor
      DItsz = DTtsz * factor
      DIcib = DTcib * factor
      DIradiops = DTradiops * factor

      # Rayleigh-Jeans temperatures [muKrj], as a function of freq
      # convert from muK to muKrj
      factor = self.convertIntSITo(self.Nu, kind="tempKrj")
      factor /= self.convertIntSITo(self.Nu, kind="tempKcmb")
      #
      DTcmbRJ = DTcmb * factor
      DTkszRJ = DTksz * factor
      DTtszRJ = DTtsz * factor
      DTcibRJ = DTcib * factor
      DTradiopsRJ = DTradiops * factor
      
   
      # Intensity [Jy/sr]
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.axhline(0.)
      ax.plot(self.Nu/1.e9, DIcmb, label=r'CMB')
      ax.plot(self.Nu/1.e9, DIksz, label=r'kSZ')
      ax.plot(self.Nu/1.e9, DItsz, label=r'tSZ')
      ax.plot(self.Nu/1.e9, DIcib, label=r'CIB')
      ax.plot(self.Nu/1.e9, DIradiops, label=r'Radio PS')
      #
      ax.legend(loc=1)
      ax.set_xlim((0., 1.e3))
      ax.set_ylim((-1.e3, 4.e3))
      ax.set_xlabel(r'$\nu$ [GHz]')
      ax.set_ylabel(r'$\delta I_\text{RMS}$ [Jy/sr]')
   
   
      # Intensity [Jy/sr], log-log
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.axhline(0.)
      ax.plot(self.Nu/1.e9, DIcmb, label=r'CMB')
      ax.plot(self.Nu/1.e9, DIksz, label=r'kSZ')
      ax.plot(self.Nu/1.e9, DItsz, 'g', label=r'tSZ')
      ax.plot(self.Nu/1.e9, -DItsz, 'g--')
      ax.plot(self.Nu/1.e9, DIcib, label=r'CIB')
      ax.plot(self.Nu/1.e9, DIradiops, label=r'Radio PS')
      #
      ax.legend(loc=2)
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlim((1., 1.e4))
      ax.set_ylim((1., 1.e6))
      ax.set_xlabel(r'$\nu$ [GHz]')
      ax.set_ylabel(r'$\delta I_\text{RMS}$ [Jy/sr]')


      # Temperature [muKcmb]
      fig=plt.figure(2)
      ax=fig.add_subplot(111)
      #
      ax.axhline(0.)
      ax.plot(self.Nu/1.e9, DTcmb, label=r'CMB')
      ax.plot(self.Nu/1.e9, DTksz, label=r'kSZ')
      ax.plot(self.Nu/1.e9, DTtsz, 'g', label=r'tSZ')
      ax.plot(self.Nu/1.e9, -DTtsz, 'g--')
      ax.plot(self.Nu/1.e9, DTcib, label=r'CIB')
      ax.plot(self.Nu/1.e9, DTradiops, label=r'Radio PS')
      #
      ax.legend(loc=1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.set_ylim((5.e-3, 1.e4))
      ax.set_xlabel(r'$\nu$ [GHz]')
      ax.set_ylabel(r'$\delta T_\text{RMS}$ [$\mu$K$_\text{CMB}$]')


      # Temperature [muKrj]
      fig=plt.figure(3)
      ax=fig.add_subplot(111)
      #
      ax.axhline(0.)
      ax.plot(self.Nu/1.e9, DTcmbRJ, label=r'CMB')
      ax.plot(self.Nu/1.e9, DTkszRJ, label=r'kSZ')
      ax.plot(self.Nu/1.e9, DTtszRJ, 'g', label=r'tSZ')
      ax.plot(self.Nu/1.e9, -DTtszRJ, 'g--')
      ax.plot(self.Nu/1.e9, DTcibRJ, label=r'CIB')
      ax.plot(self.Nu/1.e9, DTradiopsRJ, label=r'Radio PS')
      #
      ax.legend(loc=1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.set_ylim((5.e-3, 1.e4))
      ax.set_xlabel(r'$\nu$ [GHz]')
      ax.set_ylabel(r'$\delta T_\text{RMS}$ [$\mu$K$_\text{RJ}$]')


      plt.show()


   ##################################################################################
   ##################################################################################



   def rmsT(self, fcl, lMin=1., lMax=1.e4, fwhm=0.):
      '''Computes the RMS temperature fluctuation for a component with power spectrum fcl,
      between the specified lMin and lMax.
      fwhm = 0. for perfect beam
      fwhm = None for current experiment beam
      fwhm = any other value in rad
      '''
      f = lambda l: fcl(l) * l / (2.*np.pi) * self.fbeam(l, fwhm=fwhm)**2
      result, error = integrate.quad(f, lMin, lMax, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      return result


   def printRmsT(self, fwhm=0.):
      '''compute the variance of the temperature at a given point
      in muK.
      fwhm = 0. for perfect beam
      fwhm = None for current experiment beam
      fwhm = any other value in rad
      '''
      print "- temperature fluctuations due to CMB:", self.rmsT(self.flensedTT), "muK"
      # detector noise would diverge, because it is a constant divided by the beam**2
      print "- temperature fluctuations due to CIB:", self.rmsT(self.fCIB), "muK"
      print "- temperature fluctuations due to tSZ:", self.rmsT(self.ftSZ), "muK"
      print "- temperature fluctuations due to kSZ:", self.rmsT(self.fkSZ), "muK"
      
      
   def fsigmaMatchedFilter(self, fprofile=None, ftotalTT=None, lMin=None, lMax=None):
      '''outputs the uncertainty on amplitude of profile
      given the total power in the map
      fprofile: isotropic profile (before beam convolution)
      if none, use the beam as profile (ie point source)
      If temperature map in muK, then output in muK*sr
      If temperature map in Jy/sr, then output in Jy
      '''
      if ftotalTT is None:
         ftotalTT = self.ftotalTT
      if fprofile is None:
         f = lambda l: l/(2.*np.pi) / ftotalTT(l)
      else:
         f = lambda l: l/(2.*np.pi) * fprofile(l) / ftotalTT(l)
      if lMin is None:
         lMin = self.lMin
      if lMax is None:
         lMax = self.lMaxT
      result = integrate.quad(f, lMin, lMax, epsabs=0., epsrel=1.e-3)[0]
      result = 1./np.sqrt(result)
      return result



   ##################################################################################
   ##################################################################################


   def genWiggleNoWiggle(self, test=False):
      """Create a wiggle-only and a no-wiggle CMB unlensed power spectrum,
      By doing a smooth interpolation. A bit of an art...
      """

      L = np.linspace(10., 1.e4, 2001)
      ClCmb = np.array(map(self.funlensedTT, L))

      # no-wiggle power spectrum
      forUnlensedTTNoWiggle = UnivariateSpline(np.log(L), np.log(ClCmb), k=3, s=20., ext='const')
      funlensedTTNoWiggle = lambda l: np.exp(forUnlensedTTNoWiggle(np.log(l)))
      ClCmbNoWiggle = np.array(map(funlensedTTNoWiggle, L))

      # wiggle-only power spewctrum
      forUnlensedTTWiggleOnly = UnivariateSpline(np.log(L), ClCmb-ClCmbNoWiggle, k=3, s=0., ext='const')
      funlensedTTWiggleOnly = lambda l: forUnlensedTTWiggleOnly(np.log(l))
      ClCmbWiggleOnly = np.array(map(funlensedTTWiggleOnly, L))

      if test:
         
         fig=plt.figure(0)
         ax=fig.add_subplot(111)
         #
         ax.plot(L, L*(L+1.)/(2.*np.pi)*ClCmb, 'k', label=r'true')
         ax.plot(L, L*(L+1.)/(2.*np.pi)*ClCmbNoWiggle, 'b', label=r'no-wiggle')
         ax.plot(L, L*(L+1.)/(2.*np.pi)*ClCmbWiggleOnly, 'r', label=r'wiggle-only')
         ax.plot(L, -L*(L+1.)/(2.*np.pi)*ClCmbWiggleOnly, 'r--')
         #
         ax.legend()
         ax.set_xscale('log')
         ax.set_yscale('log')
         ax.set_xlabel(r'$\ell$')
         ax.set_ylabel(r'$C_\ell$')

         plt.show()


         fig=plt.figure(1)
         ax=fig.add_subplot(111)
         #
         ax.plot(L, ClCmb/ClCmb, 'k')
         ax.plot(L, ClCmb/ClCmbNoWiggle, 'b')
         #
         ax.set_xscale('log')
         #ax.set_yscale('log')
         ax.set_xlabel(r'$\ell$')
         ax.set_ylabel(r'$C_\ell / C_\ell^\text{no wiggle}$')

         plt.show()

      return funlensedTTNoWiggle, funlensedTTWiggleOnly


   ##################################################################################
   ##################################################################################
   # Plot power spectra

   def plotCl(self):
      '''Show power spectra of TT, EE, TE, BB, debeamed.
      '''
      Nl = 1001
      L = np.logspace(np.log10(1.), np.log10(3.6e4), Nl, 10.)

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      f = 1. / self.fdl_to_cl(L)
      #
      ax.loglog(L, f * self.flensedTT(L), 'k', lw=2, label=r'TT')
      ax.loglog(L, f * self.flensedEE(L), 'b', lw=2, label=r'EE')
      ax.loglog(L, f * self.flensedTE(L), 'r', lw=2, label=r'TE')
      ax.loglog(L, -f * self.flensedTE(L), 'r--', lw=2)
      ax.loglog(L, f * self.flensedBB(L), 'g', lw=2, label=r'BB')
      #
      ax.loglog(L, f * self.ftotalTT(L), 'k', lw=1)
      ax.loglog(L, f * self.ftotalEE(L), 'b', lw=1)
      ax.loglog(L, f * self.ftotalTE(L), 'r', lw=1)
      ax.loglog(L, -f * self.ftotalTE(L), 'r--', lw=1)
      ax.loglog(L, f * self.ftotalBB(L), 'g', lw=1)
      #
      ax.grid()
      ax.legend(loc=1)
      #ax.set_xlim((1.e2, 1.e4))
      ax.set_ylim((1.e-6, 1.e5))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell(\ell+1)\; C_\ell \; /(2\pi)$ [$(\mu K_\text{CMB})^2$]')

      plt.show()


   def plotClTT(self):
      '''Show various components in TT power spectrum, debeamed.
      '''
      Nl = 1001
      L = np.logspace(np.log10(1.), np.log10(3.6e4), Nl, 10.)


      
      fig=plt.figure(0, figsize=(12, 8))
      ax=plt.subplot(111)
      #
      f = 1. / self.fdl_to_cl(L)
      #
      ax.loglog(L, f * self.flensedTT(L), 'r', lw=2, label=r'CMB')
      ax.loglog(L, f * self.fCIB(L), 'b', lw=2, label=r'CIB')
      ax.loglog(L, f * self.fCIBPoisson(L), 'b--', lw=1, label=r'CIB poisson')
      ax.loglog(L, f * self.fCIBClustered(L), 'b--', lw=1, label=r'CIB clustered')
      ax.loglog(L, f * self.ftSZ(L), 'g', lw=2, label=r'tSZ')
      ax.loglog(L, f * self.fkSZ(L), 'g--', lw=2, label=r'kSZ')
      ax.loglog(L, -f * self.ftSZ_CIB(L), 'm', lw=2, label=r'$|$ tSZ x CIB $|$')
      ax.loglog(L, f * self.fradioPoisson(L), 'y', lw=2, label=r'radio Poisson')
      ax.loglog(L, f * self.fgalacticDust(L), 'r', lw=2, label=r'galactic dust')
      ax.loglog(L, f * self.fdetectorNoise(L), 'k--', lw=2, label=r'detector noise')
      ax.loglog(L, f * self.ftotalTT(L), 'k', lw=2, label=r'total')
      #
      ax.grid()
      ax.legend(loc=2)
      ax.set_xlim((100., 2.4e4))
      ax.set_ylim((1.e-4, 1.e6))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell(\ell+1)\; C_\ell \; /(2\pi)$ [$(\mu K_\text{CMB})^2$]')

      plt.show()


###############################################################################
###############################################################################
# Examples
'''
# ACT 148GHz
act148cmb = CMB(beam=1.4, noise=12., nu1=148.e9, nu2=148.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, atm=False, name="act148")

# Planck SMICA map
planckSmicaCmb = CMB(beam=5., noise=60., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, atm=False, name="plancksmica")

# ACTPol
actpolCmb = CMB(beam=1.4, noise=18., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, atm=False, name="actpol")

# AdvACT
advactCmb = CMB(beam=1.4, noise=10., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, atm=False, name="actpol")

# CMB S4
cmbs4 = CMB(beam=1., noise=1., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, atm=False, name="cmbs4")

# the "reference CMB experiment" from Hu Okamoto 2002
huokamoto02 = CMB(beam=4., noise=1., nu1=143.e9, nu2=143.e9, lMin=1., lMaxT=1.e4, lMaxP=1.e4, atm=False, name="cmbs4")
'''

###############################################################################
###############################################################################
# CIB

class CIB(CMB):
   
   def __init__(self, beam=1., noise=1., nu1=143.e9, nu2=143.e9, lMin=30., lMaxT=3.e3, lMaxP=5.e3, name=None):
      # name
      if name is None:
         self.name = "cib_nu"+str(int(nu1/1.e9))+"_nu"+str(int(nu2/1.e9))+"_beam"+str(round(beam, 3))+"_noise"+str(round(noise, 3))+"_lmin"+str(int(lMin))+"_lmaxT"+str(int(lMaxT))+"_lmaxP"+str(int(lMaxP))
      else:
         self.name = name
      # frequencies in Hz (irrelevant)
      self.nu1 = nu1
      self.nu2 = nu2
      # beam fwhm in radians
      self.fwhm = beam * (np.pi/180.)/60.
      # detector sensitivity in muK*rad.
      self.sensitivity = noise * (np.pi/180.)/60.
      # ell limits
      self.lMin = lMin
      self.lMaxT = lMaxT
      self.lMaxP = lMaxP
      
      super(CIB, self).__init__()
      
      # do not include the foregrounds in the total (cleaned map)
      self.funlensedTT = lambda l: self.fCIB(l, nu1, nu2)
      self.funlensedEE = lambda l: 0.
      self.funlensedBB = lambda l: 0.
      self.funlensedTE = lambda l: 0.
      #
      self.ftotalTT = lambda l: self.fCIB(l, nu1, nu2) + self.fdetectorNoise(l)
      self.ftotalEE = lambda l: 0. + 2.*self.fdetectorNoise(l)
      self.ftotalBB = lambda l: 0. + 2.*self.fdetectorNoise(l)
      self.ftotalTE = lambda l: 0.


###############################################################################
###############################################################################
# !!!!!!!!! Incomplete
# CIB: fit to the auto-spectrum of Planck 15 GNILC maps, from Simo
# beam is 5arcmin for all frequencies
# the noise is incorrect here
# unit here is MJy/sr for the power spectrum

class CIBPlanck15FitSimo(CMB):
   
   def __init__(self, beam=5., noise=1., nu1=143.e9, nu2=143.e9, lMin=30., lMaxT=3.e3, lMaxP=5.e3):
      # name
      self.name = "cibplanckfit_nu"+str(int(nu1/1.e9))+"_nu"+str(int(nu2/1.e9))+"_beam"+str(round(beam, 3))+"_noise"+str(round(noise, 3))+"_lmin"+str(int(lMin))+"_lmaxT"+str(int(lMaxT))+"_lmaxP"+str(int(lMaxP))
      # frequencies in Hz (irrelevant)
      self.nu1 = nu1
      self.nu2 = nu2
      # beam fwhm in radians
      self.fwhm = beam * (np.pi/180.)/60.
      # detector sensitivity in muK*rad.
      self.sensitivity = noise * (np.pi/180.)/60.
      # ell limits
      self.lMin = lMin
      self.lMaxT = lMaxT
      self.lMaxP = lMaxP
      
      super(CIB, self).__init__()
      
      # do not include the foregrounds in the total (cleaned map)
      self.funlensedTT = lambda l: self.fCIB(l, nu1, nu2)
      self.funlensedEE = lambda l: 0.
      self.funlensedBB = lambda l: 0.
      self.funlensedTE = lambda l: 0.
      #
      self.ftotalTT = lambda l: self.fCIB(l, nu1, nu2) + self.fdetectorNoise(l)
      self.ftotalEE = lambda l: 0. + 2.*self.fdetectorNoise(l)
      self.ftotalBB = lambda l: 0. + 2.*self.fdetectorNoise(l)
      self.ftotalTE = lambda l: 0.


   ###############################################################################
   # CIB Poisson and clustered

   def fCIBPoisson(self, l, nu1=None, nu2=None):
      a_CIBP = 7.0
      Td = 9.7
      betaP = 2.1
      if nu1 is None:
         nu1 = self.nu1
      if nu2 is None:
         nu2 = self.nu2
         return a_CIBP/a * (l/3000.)**2 * self.mu(nu1, betaP, Td)*self.mu(nu2, betaP, Td)/self.mu(150.e9, betaP, Td)**2 * self.fdl_to_cl(l)

   def fCIBClustered(self, l, nu1=None, nu2=None):
      a_CIBC = 5.7
      n = 1.2
      Td = 9.7
      betaC = 2.1
      if nu1 is None:
         nu1 = self.nu1
      if nu2 is None:
         nu2 = self.nu2
         return a_CIBC * (l/3000.)**(2-n) * self.mu(nu1, betaC, Td)*self.mu(nu2, betaC, Td)/self.mu(150.e9, betaC, Td)**2 * self.fdl_to_cl(l)
   
   def fCIB(self, l, nu1=None, nu2=None):
      return self.fCIBPoisson(l, nu1, nu2) + self.fCIBClustered(l, nu1, nu2)

###############################################################################
###############################################################################
# CIB halo model, tabulated
# sent to me by Hao-Yi Wu at NORDITA conference, summer 2017
# from Wu Dore 2017
# units are Jy^2/sr for the power spectra

class CIBWuDore17(CMB):
   
   def __init__(self, beam=1., noise=1., nu1=143.e9, nu2=143.e9, lMin=30., lMaxT=3.e3, lMaxP=5.e3):
      # name
      #      self.name = "cmbs4"
      self.name = "cib_wudore17_nu"+str(int(nu1/1.e9))+"_nu"+str(int(nu2/1.e9))+"_beam"+str(round(beam, 3))+"_noise"+str(round(noise, 3))+"_lmin"+str(int(lMin))+"_lmaxT"+str(int(lMaxT))+"_lmaxP"+str(int(lMaxP))
      # frequencies in Hz
      self.nu1 = nu1
      self.nu2 = nu2
      # convert beam fwhm from arcmin to rad
      self.fwhm = beam * (np.pi/180.)/60.
      # noise assumed to be in Jy / rad
      self.sensitivity = noise
      # ell limits
      self.lMin = lMin
      self.lMaxT = lMaxT
      self.lMaxP = lMaxP
      
      super(CIBWuDore17, self).__init__()
      
      # load tabulated spectra from Wu Dore 2017
      # shot noises in Jy/sr, for freq in GHz
      self.shotNoises = {217: 13.5556, 353:228.754, 545: 1796.18, 857: 7379.63}
      self.loadTabulatedCIB(nu1, nu2)
      
      # do not include the foregrounds in the total (cleaned map)
      self.funlensedTT = lambda l: self.fCIB(l)
      self.funlensedEE = lambda l: 0.
      self.funlensedBB = lambda l: 0.
      self.funlensedTE = lambda l: 0.
      #
      self.ftotalTT = lambda l: self.fCIB(l) + self.fdetectorNoise(l)
      self.ftotalEE = lambda l: 0. + 2.*self.fdetectorNoise(l)
      self.ftotalBB = lambda l: 0. + 2.*self.fdetectorNoise(l)
      self.ftotalTE = lambda l: 0.


   def loadTabulatedCIB(self, nu1, nu2):
      # read the data file
      nu1, nu2 = np.sort([nu1, nu2])
      path = "./input/cib_wu_dore_17/CL_1h2h_"+str(int(nu1/1.e9))+"x"+str(int(nu2/1.e9))+"_base.dat"
      data = np.genfromtxt(path)
      
      # interpolate the 1h, 2h and shot noise
      self.flnCIB_1hln = interp1d(np.log(data[:,0]), np.log(data[:,1]), kind='linear', bounds_error=False, fill_value='extrapolate')
      self.fCIB_1h = lambda l: np.exp(self.flnCIB_1hln(np.log(l)))
      #
      self.flnCIB_2hln = interp1d(np.log(data[:,0]), np.log(data[:,2]), kind='linear', bounds_error=False, fill_value='extrapolate')
      self.fCIB_2h = lambda l: np.exp(self.flnCIB_2hln(np.log(l)))
      #
      if (nu1==nu2):
         self.fCIB_shot = lambda l: self.shotNoises[int(nu1/1.e9)]# *(l>=data[0,0])*(l<=data[-1,0])
      else:
         self.fCIB_shot = lambda l: 0.
      #
      self.fCIB = lambda l: self.fCIB_1h(l) + self.fCIB_2h(l) + self.fCIB_shot(l)


   def plot(self):
      # check the interpolation
      L = np.logspace(np.log10(1.), np.log10(1.e5), 1001, 10.)
      Cl_1h = np.array(map(self.fCIB_1h, L))
      Cl_2h = np.array(map(self.fCIB_2h, L))
      Cl_shot = np.array(map(self.fCIB_shot, L))
      Cl = np.array(map(self.fCIB, L))
      
      # superimpose the data points to check
      nu1, nu2 = np.sort([self.nu1, self.nu2])
      path = "./input/cib_wu_dore_17/CL_1h2h_"+str(int(nu1/1.e9))+"x"+str(int(nu2/1.e9))+"_base.dat"
      data = np.genfromtxt(path)
      

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # table from Hao-Yi Wu
      ax.loglog(data[:,0], data[:,1], 'k.')
      ax.loglog(data[:,0], data[:,2], 'k.')
      #
      # interpolated values
      ax.loglog(L, Cl, 'k', label=r'total')
      ax.loglog(L, Cl_2h, 'r', label=r'2h')
      ax.loglog(L, Cl_1h, 'b', label=r'1h')
      ax.loglog(L, Cl_shot, 'g', label=r'shot')
      #
      ax.legend(loc=1)
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$C_\ell^{'+str(int(self.nu1/1.e9))+'-'+str(int(self.nu2/1.e9))+'}$ [Jy$^2$/sr]')
      #
      fig.savefig("/Users/Emmanuel/Desktop/cib_power_wudore17.pdf", bbox_inches='tight')

      plt.show()



