"""
special stellar mass functions for DESCQA project
"""

from __future__ import (division, print_function, absolute_import)

import numpy as np
from scipy.interpolate import interp1d
from astropy.modeling.models import custom_model

__all__ = ['LiWhite_2009_phi', 'MBII_Desc_Phi', 'Illustris_Desc_Phi']
__author__ = ['Duncan Campbell']

class LiWhite_2009_phi(object):
    """
    stellar mass function based on Li & White 2009, arXiv:0901.0706
    """
    def __init__(self):
        """
        """
        
        self.publication = ['arXiv:0901.0706']
        
        self.littleh = 1.0
        
        #parameters from table #1
        self.min_mstar1 = 8.0
        self.phi1 = 0.01465
        self.x1 = 9.6124
        self.alpha1 = -1.1309
        self.max_mstar1 = 9.33
        
        self.min_mstar2 = 9.33
        self.phi2 = 0.01327
        self.x2 = 10.3702
        self.alpha2 = -0.9004
        self.max_mstar2 = 10.67
        
        self.min_mstar3 = 10.67
        self.phi3 = 0.0044
        self.x3 = 10.7104
        self.alpha3 = -1.9918
        self.max_mstar3 = 12.0
        
        #used to build piecewise function
        @custom_model
        def interval(x,x1=0.0,x2=1.0):
            """
            return 1 if x is in the range (x1,x2] and 0 otherwise
            """
            x = np.array(x)
            mask = ((x<=x2) & (x>x1))
            result = np.zeros(len(x))
            result[mask]=1.0
            return result
        
        #define components of double Schechter function
        s1 = Log_Schechter(phi0=self.phi1, x0=self.x1, alpha=self.alpha1)*interval(x1=-np.inf,x2=self.max_mstar1)
        s2 = Log_Schechter(phi0=self.phi2, x0=self.x2, alpha=self.alpha2)*interval(x1=self.min_mstar2,x2=self.max_mstar2)
        s3 = Log_Schechter(phi0=self.phi3, x0=self.x3, alpha=self.alpha3)*interval(x1=self.min_mstar3,x2=np.inf)
        
        #create piecewise model
        self.s = s1 + s2 + s3
        
    
    def __call__(self, mstar, **kwargs):
        """
        stellar mass function from Li & White 2009, arXiv:0901.0706
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        #take log of stellar masses
        mstar = np.log10(mstar) - 0.136
        
        return self.s(mstar)

class MBII_Desc_Phi(object):
    """
    stellar mass function of all galaxies in MBII
    """
    
    def __init__(self, **kwargs):
        """
        intialize stellar mass function
        """
        
        #tabulated stellar masses (h^-2 Msol)
        mstar = [1.412537544622755423e+07, 2.818382931264449283e+07, 5.623413251903490722e+07,
                 1.122018454301965237e+08, 2.238721138568337858e+08, 4.466835921509634852e+08,
                 8.912509381337441206e+08, 1.778279410038922787e+09, 3.548133892335760593e+09,
                 7.079457843841373444e+09, 1.412537544622755432e+10, 2.818382931264449310e+10,
                 5.623413251903490448e+10, 1.122018454301965332e+11, 2.238721138568337708e+11,
                 4.466835921509635010e+11, 8.912509381337440186e+11]
        #tabulated number densities (h^3 Mpc^-3 dex^-1)
        phi = [3.525666666666668614e-01, 4.933833333333336735e-01, 5.379400000000003068e-01,
               4.469533333333336467e-01, 3.189300000000002133e-01, 2.035066666666668078e-01,
               1.047366666666667140e-01, 5.000666666666669230e-02, 2.607000000000001330e-02,
               1.307333333333334131e-02, 7.673333333333337558e-03, 4.313333333333335684e-03,
               2.320000000000001315e-03, 1.120000000000000552e-03, 5.400000000000003322e-04,
               2.633333333333334639e-04, 9.666666666666672144e-05]
        mstar = np.array(mstar)
        phi = np.array(phi)
        
        #create interpolation function
        logf = interp1d(np.log10(mstar), np.log10(phi),fill_value='extrapolate')
        
        self.s = lambda x: 10.0**logf(np.log10(x))
    
    def __call__(self, mstar, **kwargs):
        """
        stellar mass function
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        return self.s(mstar)

class Illustris_Desc_Phi(object):
    """
    stellar mass function of all galaxies in Illustris
    """
    
    def __init__(self, **kwargs):
        """
        intialize stellar mass function
        """
        
        #tabulated stellar masses (h^-2 Msol)
        mstar = [1.412537544622755423e+07, 2.818382931264449283e+07, 5.623413251903490722e+07,
                 1.122018454301965237e+08, 2.238721138568337858e+08, 4.466835921509634852e+08,
                 8.912509381337441206e+08, 1.778279410038922787e+09, 3.548133892335760593e+09,
                 7.079457843841373444e+09, 1.412537544622755432e+10, 2.818382931264449310e+10,
                 5.623413251903490448e+10, 1.122018454301965332e+11, 2.238721138568337708e+11,
                 4.466835921509635010e+11, 8.912509381337440186e+11]
        #tabulated number densities (h^3 Mpc^-3 dex^-1)
        phi = [2.054004938271606073e-01, 1.815308641975309500e-01, 1.543269135802469993e-01,
               1.250528395061729081e-01, 1.013965432098766078e-01, 8.185679012345684069e-02,
               6.901728395061731969e-02, 4.751802469135805312e-02, 3.247407407407409269e-02,
               2.201283950617285295e-02, 1.364543209876543928e-02, 8.778271604938277239e-03,
               5.064691358024694598e-03, 2.315061728395062784e-03, 1.003456790123457339e-03,
               2.923456790123458689e-04, 6.320987654320990698e-05]
        
        mstar = np.array(mstar)
        phi = np.array(phi)
        
        #create interpolation function
        logf = interp1d(np.log10(mstar), np.log10(phi), fill_value='extrapolate')
        
        self.s = lambda x: 10.0**logf(np.log10(x))
    
    def __call__(self, mstar):
        """
        stellar mass function
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        return self.s(mstar)

@custom_model
def Log_Schechter(x, phi0=0.001, x0=10.5, alpha=-1.0):
    """
    log schecter function
    """
    x = np.asarray(x)
    x = x.astype(float)
    norm = np.log(10.0)*phi0
    val = norm*(10.0**((x-x0)*(1.0+alpha)))*np.exp(-10.0**(x-x0))
    return val
