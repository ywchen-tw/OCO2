"""
Reader and processing module for ERT OCO files
pre-release
"""

import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import numpy as np
from sys import exit as ext

class OCOSIM:
    def __init__(self,file):
        self.file=file
        with h5py.File(file, "r") as h1:
            self.lat  = h1['lat'][...] 
            self.lon  = h1['lon'][...] 
            self.co2  = h1['xco2'][...]
            self.l1b  = h1['rad_oco'][...]
            self.wvl  = h1['wvl_oco'][...]
            # calculations mapped onto OCO soundings
            self.lam  = h1['wvl_mca'][...]
            self.clr  = h1['rad_mca_ipa0'][...] # no clouds
            self.c3d  = h1['rad_mca_3d'][...] # clouds, 3D
            self.clrs = h1['rad_mca_ipa0_std'][...] # same as above, but standard deviation
            self.c3ds = h1['rad_mca_3d_std'][...]
            # radiance field for full domain
            self.rad  = h1['rad_mca_3d_domain'][...]
            self.dom  = h1['extent_domain'][...] # lats/lons
            self.snd  = h1['snd_id'][...]
            self.toa  = h1['toa'][...]
            self.sza  = h1['sza_mca'][...]
            self.sfc_alb = h1['sfc_alb'][...]
            self.sza_avg  = h1['sza_avg'][...]
            self.np   = h1['Np'][...]
            self.logic= h1['logic'][...]
            self.nz,self.nf=self.lat.shape
            if 'sfc_pres' in h1.keys():
                self.psur = h1['sfc_pres'][...]
            else:
                self.psur = np.zeros([self.nz,8])
            if 'rad_mca_ipa0_domain' in h1.keys():
                self.lon2d = h1['lon2d'][...]
                self.lat2d = h1['lat2d'][...]
                self.rad_clr = h1['rad_mca_ipa0_domain'][...]
                self.rad_c3d = h1['rad_mca_3d_domain'][...]
                self.rad_clrs = h1['rad_mca_ipa0_domain_std'][...]
                self.rad_c3ds = h1['rad_mca_3d_domain_std'][...]
                self.cld_position = h1['rad_mca_ipa0_domain'][...]
                self.cld_position[...] = np.nan
                clr_all = self.rad_clr.copy()
                flt_all = np.where(clr_all==0)
                clr_all[flt_all[0],flt_all[1],flt_all[2]] = 100000000000
                self.sl_all  = (self.rad_c3d-self.rad_clr) / clr_all        # S_lamda
                self.sls_all = (self.rad_c3ds/clr_all + self.rad_clrs/clr_all)  # S_lamda standard deviation
            
            clr = self.clr.copy()
            flt = np.where(clr==0)
            clr[flt[0], flt[1], flt[2]] = 100000000000
            
            self.sl  = (self.c3d-self.clr) / clr        # S_lamda
            self.sls = (self.c3ds/clr + self.clrs/clr)  # S_lamda standard deviation
        
    def get_slope(self,fp,z,mode='unperturb'):
        nwl=self.sls[z,fp,:].shape[0]
        flt=np.where(self.sls[z,fp,:]>1e-6)
        use=len(flt[0])
        if use==nwl:
            w=1./self.sls[z,fp,:]    
            if mode=='unperturb':
                x=self.c3d[z,fp,:]/self.toa[:]*np.pi
            else:
                x=self.clr[z,fp,:]/self.toa[:]*np.pi
            res=np.polyfit(x,self.sl[z,fp,:],1,w=w,cov=True) # now get covariance as well!
            slope,intercept=res[0]
            slopestd=np.sqrt(res[1][0][0])
            interceptstd=np.sqrt(res[1][1][1])
        else:
            slope=np.nan; slopestd=np.nan; intercept=np.nan; interceptstd=np.nan
        return(slope,slopestd,intercept,interceptstd)
    
    def get_all_slope(self,i,j,mode='unperturb'):
        nwl=self.sls_all[i,j,:].shape[0]
        flt=np.where(self.sls_all[i,j,:]>1e-6)
        use=len(flt[0])
        if use==nwl:
            w=1./self.sls_all[i,j,:]    
            if mode=='unperturb':
                x=self.rad_c3d[i,j,:]/self.toa[:]*np.pi
            else:
                x=self.rad_clr[i,j,:]/self.toa[:]*np.pi
            res=np.polyfit(x,self.sl_all[i,j,:],1,w=w,cov=True) # now get covariance as well!
            slope,intercept=res[0]
            slopestd=np.sqrt(res[1][0][0])
            interceptstd=np.sqrt(res[1][1][1])
        else:
            slope=np.nan; slopestd=np.nan; intercept=np.nan; interceptstd=np.nan
        return(slope,slopestd,intercept,interceptstd)
        
    def slopes(self,fp,mode='unperturb'): # goes through entire line for a given footprint fp
        if not hasattr(self,'slope'):
            self.slope=np.zeros([self.nz,self.nf,2])
            self.inter=np.zeros([self.nz,self.nf,2])
        for z in range(self.nz):
            slope,slopestd,inter,interstd=self.get_slope(fp,z,mode='unperturb')
            self.slope[z,fp,:]=[slope,slopestd]
            self.inter[z,fp,:]=[inter,interstd]
        if hasattr(self, 'lon2d'):
            if not hasattr(self,'slope_all'):
                self.slope_all=np.zeros([self.lon2d.shape[0],self.lon2d.shape[1],2])
                self.inter_all=np.zeros([self.lon2d.shape[0],self.lon2d.shape[1],2])
            for lon in range(self.lon2d.shape[0]):
                for lat in range(self.lon2d.shape[1]):
                    slope_all,slopestd_all,inter_all,interstd_all=self.get_all_slope(lon,lat,mode='unperturb')
                    self.slope_all[lon,lat,:]=[slope_all,slopestd_all]
                    self.inter_all[lon,lat,:]=[inter_all,interstd_all]  
