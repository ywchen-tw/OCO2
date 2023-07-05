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
        h1  = h5py.File(file, "r")
        self.lat  = h1['lat'][...] 
        self.lon  = h1['lon'][...] 
        self.co2  = h1['xco2'][...]
        self.l1b  = h1['rad_oco'][...]
        self.wvl  = h1['wvl_oco'][...]
        # calculations mapped onto OCO soundings
        self.lam  = h1['wvl_mca'][...]
        self.clr  = h1['rad_mca_ipa0'][...] # no clouds
        self.c1d  = h1['rad_mca_ipa'][...] # clouds, but 1D
        self.c3d  = h1['rad_mca_3d'][...] # clouds, 3D
        self.clrs = h1['rad_mca_ipa0_std'][...] # same as above, but standard deviation
        self.c1ds = h1['rad_mca_ipa_std'][...] 
        self.c3ds = h1['rad_mca_3d_std'][...]
        # radiance field for full domain
        self.rad  = h1['rad_mca_3d_domain'][...]
        self.dom  = h1['extent_domain'][...] # lats/lons
        self.snd  = h1['snd_id'][...]
        self.toa  = h1['toa'][...]
        self.mu  = h1['sza_mca'][...]
        self.np   = h1['Np'][...]
        self.logic= h1['logic'][...]
        self.nz,self.nf=self.lat.shape
        if 'sfc_pres' in h1.keys():
            self.psur = h1['sfc_pres'][...]
        else:
            self.psur = np.zeros([self.nz,8])
        if 'rad_mca_ipa_domain' in h1.keys():
            self.lon2d = h1['lon2d'][...]
            self.lat2d = h1['lat2d'][...]
            self.rad_clr = h1['rad_mca_ipa0_domain'][...]
            self.rad_c1d = h1['rad_mca_ipa_domain'][...]
            self.rad_c3d = h1['rad_mca_3d_domain'][...]
            self.rad_clrs = h1['rad_mca_ipa0_domain_std'][...]
            self.rad_c1ds = h1['rad_mca_ipa_domain_std'][...]
            self.rad_c3ds = h1['rad_mca_3d_domain_std'][...]
            self.cld_position = h1['rad_mca_ipa0_domain'][...]
            self.cld_position[...] = np.nan
            clr_all=self.rad_clr.copy()
            flt_all=np.where(clr_all==0)
            clr_all[flt_all[0],flt_all[1],flt_all[2]]=100000000000
            self.sl_all  = (self.rad_c3d-self.rad_clr) / clr_all        # S_lamda
            self.sls_all = (self.rad_c3ds/clr_all + self.rad_clrs/clr_all)  # S_lamda standard deviation
        
        clr=self.clr.copy()
        flt=np.where(clr==0)
        clr[flt[0],flt[1],flt[2]]=100000000000
        
        self.sl  = (self.c3d-self.clr) / clr        # S_lamda
        self.sls = (self.c3ds/clr + self.clrs/clr)  # S_lamda standard deviation
        h1.close()
        
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
            
    def visualize_rad(self,l=10,fp=0):   
        f=np.pi/self.toa[l]
        fig,ax = plt.subplots()
        ax.plot(self.c3d[:,fp,l]*f,'k-',label='3D reflectance')
        ax.plot(self.c1d[:,fp,l]*f,'k:',label='1D reflectance')
        ax.plot(self.clr[:,fp,l]*f,'bo',label='clr reflectance')
        ax.set_ylim(0,1)
        ax.set_xlabel('row #')
        ax.set_ylabel('Reflectance')
        ax.legend()
        ax.set_title('fp='+str(fp)+' $\lambda$='+str(l))
        a1=ax.twinx()
        a1.plot(self.psur[:,fp]*0.01,'ro')
        a1.set_ylabel('P sur [mbar]',color='red')
        a2=ax.twinx()
        #for fp0 in range(8):
        #    print(fp0)
        a2.plot(self.co2[:,fp]*1e6,'go')
        a2.spines["right"].set_position(("axes", 1.2))
        a2.set_ylabel('XCO2 [ppm]',color='green')
        fig.subplots_adjust(right=0.75)

    def visualize_co2(self,frame=None,l=10,s=8):
        if frame is None:
            f,frame=plt.subplots()
            
        #get mean and std dev
        cl   = self.co2.copy()
        #get rid of data beyond normal limits
        flt=np.where(np.isnan(cl))
        flo=np.where(~np.isnan(cl))
        clm  = np.median(cl[flo])
        clstd= np.std(cl[flo])
        c0   = clm-1*clstd
        c1   = clm+1*clstd
        cl[flt]=-1
        f0=np.where(cl[flo] < c0)
        cl[flo][f0]=c0
        f1=np.where(cl[flo] > c1)
        cl[flo][f1]=c1
        # plot
        fff=np.where((self.lat > self.dom[2]) & (self.lat < self.dom[3]) & (cl>0))
        cm = plt.cm.get_cmap('jet')
        frame.scatter(self.lon[fff],self.lat[fff],s=s,c=cl[fff],cmap=cm,alpha=0.9,linestyle='None')
        #plt.autoscale(enable=True)    
        return(c0,c1)
        
    def visualize_inp(self,lat,lon,inp,c0=None,c1=None,frame=None,s=8):
        if frame is None:
            f,frame=plt.subplots()
        
        #get mean and std dev
        cl   = inp
        
        if (c0 is None) | (c1 is None):
            cm=np.mean(cl)
            cs=np.std(cl)
            c0=cm-cs
            c1=cm+cs
        #get rid of data beyond normal limits
        f0=np.where(cl < c0)
        cl[f0]=c0
        f1=np.where(cl > c1)
        cl[f1]=c1
        # plot
        fff=np.where((lat > self.dom[2]) & (lat < self.dom[3]))
        cm = plt.cm.get_cmap('rainbow')
        frame.scatter(lon[fff],lat[fff],s=s,c=cl[fff],cmap=cm,alpha=0.9,linestyle='-',edgecolor='white')
        return(c0,c1)
        #plt.autoscale(enable=True)    

    def visualize_snds(self,snd,color_item=0,frame=None):
        items=['intercept O2A','slope O2A','intercept WCO2','slope WCO2','intercept SCO2','slope SCO2']
        if frame is None:
            f,frame=plt.subplots()
        cm = plt.cm.get_cmap('jet')
        lat=[]; lon=[]; clr=[]
        for snd0 in snd.keys():
            f0,f1=np.where(snd0==self.snd)
            #print(self.snd[f0,f1][0],snd0)
            lat.append(self.lat[f0,f1][0]); lon.append(self.lon[f0,f1][0])
            cl=snd[snd0][color_item]+1
            if (cl<0): cl=0
            if (cl>2): cl=2
            clr.append(cl)
        frame.scatter(lon,lat,s=25,c=clr,edgecolor='white',cmap=cm)
        #frame.set_title('color: '+items[color_item])
            
    def visualize_radfield(self,frame=None,withfp=False,withobs=False,title=False,s=None,crop=True):
        if frame is None:
            f,frame=plt.subplots()
        nl=self.toa.shape[0]
        vmin=0
        vmax=1
        clc=np.pi*self.rad.T/self.toa[nl-1]
        frame.imshow(clc, cmap='jet', origin='lower', vmin=vmin, vmax=vmax, extent=self.dom, zorder=0)
        i1=self.file.rfind('/')+1
        xlab=self.file[i1:]
        if title: frame.set_title(xlab)
        if withfp: frame.scatter(self.lon[:,:],self.lat[:,:],c=np.pi*self.c3d[:,:,nl-1]/self.toa[nl-1],s=s,vmin=vmin,vmax=vmax,cmap='jet',edgecolors='black')
        if withobs:
            idx=np.argmin(np.abs(self.wvl[0,0,:]-self.lam[nl-1]))
            obs=np.pi*self.l1b[:,:,idx]/self.toa[nl-1]
            frame.scatter(self.lon[:,:],self.lat[:,:],c=obs[:,:],s=s,vmin=vmin,vmax=vmax,cmap='jet',edgecolors='black')
        if crop:
            frame.set_xlim(self.dom[0],self.dom[1])
            frame.set_ylim(self.dom[2],self.dom[3])
            
    def plot_perturbation(self,z,fp,frame=None):
        if frame is None:
            f,frame=plt.subplots()
        frame.plot(self.c3d[z,fp,:]/self.toa*np.pi,self.sl[z,fp,:]*100,'ko')    
        for l,wl in enumerate(self.lam):
            frame.plot([self.c3d[z,fp,l]/self.toa[l]*np.pi,self.c3d[z,fp,l]/self.toa[l]*np.pi],
                     np.array([self.sl[z,fp,l]-self.sls[z,fp,l],self.sl[z,fp,l]+self.sls[z,fp,l]])*100,'k:')        
        slope,slopes         = self.slope[z,fp,:]
        intercept,intercepts = self.inter[z,fp,:]
        mn = np.min(self.c3d[z,fp,:]/self.toa[:]*np.pi)
        mx = np.max(self.c3d[z,fp,:]/self.toa[:]*np.pi)
        xx=np.linspace(mn,mx,10)
        yy=intercept+slope*xx
        y1=intercept+intercepts+(slope+slopes)*xx
        y2=intercept-intercepts+(slope-slopes)*xx
        frame.plot(xx,yy*100,'r-',linewidth=2)  
        frame.plot(xx,y1*100,'r:',linewidth=1)  
        frame.plot(xx,y2*100,'r:',linewidth=1)  
        frame.plot([0,mx],[0,0],'k--')
        # ywc added
        frame.set_title('slope: {:.4f}'.format(slope))
        frame.set_ylim(-30, 30)

    def plot_line(self,fp,l,frame=None,std=True,slope=True,withobs=False,nan=1e10):    
        if frame is None:
            f,frame=plt.subplots()
        nl=self.lam.shape[0]
        nn=np.where(self.c3d[:,fp,l]<nan)
        frame.plot(self.lat[nn[0],fp],self.c3d[nn[0],fp,l],'k-',label='3D radiance')
        if std: frame.plot(self.lat[:,fp],self.c3ds[:,fp,l],'r-',label='3DS')
        nn=np.where(self.c1d[:,fp,l]<nan)
        frame.plot(self.lat[nn[0],fp],self.c1d[nn[0],fp,l],'ko',label='1D radiance')
        nn=np.where(self.clr[:,fp,l]<nan)
        frame.plot(self.lat[nn[0],fp],self.clr[nn[0],fp,l],'bo',label='clr radiance')
        if slope:
            frame.plot(self.lat[:,fp],self.slope[:,fp,0],'g-',label='slope') # slope
            frame.plot(self.lat[:,fp],self.slope[:,fp,1],'g:',label='slope STD') # slope STD
            frame.plot(self.lat[:,fp],self.inter[:,fp,0],'y-',label='intercept') # intercept
            frame.plot(self.lat[:,fp],self.inter[:,fp,1],'y:',label='intercept STD') # intercept STD
        if withobs:
            idx=np.argmin(np.abs(self.wvl[0,0,:]-self.lam[nl-1]))
            obs=self.l1b[:,fp,idx]#/(self.toa[nl-1]*np.cos(np.pi/180.*sza))
            frame.plot(self.lat[:,fp],obs,'r:',label='observed radiance')

        frame.legend()
    
    # ywc added function
    def combine(self, OCOSIM_2):
        self.lat  = np.append(self.lat, OCOSIM_2.lat, axis=0)
        self.lon  = np.append(self.lon, OCOSIM_2.lon, axis=0)
        self.co2  = np.append(self.co2, OCOSIM_2.co2, axis=0)
        self.l1b  = np.append(self.l1b, OCOSIM_2.l1b, axis=0)
        self.wvl  = np.append(self.wvl, OCOSIM_2.wvl, axis=0)
        # calculations mapped onto OCO soundings
        self.lam  = self.lam
        self.clr  = np.append(self.clr, OCOSIM_2.clr, axis=0) # no clouds
        self.c1d  = np.append(self.c1d, OCOSIM_2.c1d, axis=0) # clouds, but 1D
        self.c3d  = np.append(self.c3d, OCOSIM_2.c3d, axis=0) # clouds, 3D
        self.clrs = np.append(self.clrs, OCOSIM_2.clrs, axis=0) # same as above, but standard deviation
        self.c1ds = np.append(self.c1ds, OCOSIM_2.c1ds, axis=0) 
        self.c3ds = np.append(self.c3ds, OCOSIM_2.c3ds, axis=0)
        # radiance field for full domain
        self.rad  = np.nan
        self.dom  = np.append(self.dom, OCOSIM_2.dom) # lats/lons
        self.dom  = self.dom.reshape((len(self.dom)//4, 4))
        self.snd  = np.append(self.snd, OCOSIM_2.snd, axis=0)
        self.toa  = self.toa
        self.np   = np.append(self.np, OCOSIM_2.np)
        self.logic= np.append(self.logic, OCOSIM_2.logic, axis=0)
        self.nz,self.nf=self.lat.shape
        self.psur = np.append(self.psur, OCOSIM_2.psur, axis=0)
        
        self.sl   = np.append(self.sl, OCOSIM_2.sl, axis=0)        # S_lamda
        self.sls  = np.append(self.sls, OCOSIM_2.sls, axis=0)  # S_lamda standard deviation
        if hasattr(self, 'slope'):
            self.slope= np.append(self.slope, OCOSIM_2.slope, axis=0)        # S_lamda
            self.inter= np.append(self.inter, OCOSIM_2.inter, axis=0)  # S_lamda standard deviation
        if hasattr(self,'slope_1km'):
            self.sl_1km   = np.append(self.sl_1km, OCOSIM_2.sl_1km, axis=0)        # S_lamda
            self.sls_1km  = np.append(self.sls_1km, OCOSIM_2.sls_1km, axis=0)  # S_lamda standard deviation
            self.slope_1km = np.append(self.slope_1km, OCOSIM_2.slope_1km, axis=0)        # S_lamda
            self.inter_1km = np.append(self.inter_1km, OCOSIM_2.inter_1km, axis=0)  # S_lamda standard deviation
            self.sl_25p   = np.append(self.sl_25p, OCOSIM_2.sl_25p, axis=0)        # S_lamda
            self.sls_25p  = np.append(self.sls_25p, OCOSIM_2.sls_25p, axis=0)  # S_lamda standard deviation
            self.slope_25p = np.append(self.slope_25p, OCOSIM_2.slope_25p, axis=0)        # S_lamda
            self.inter_25p = np.append(self.inter_25p, OCOSIM_2.inter_25p, axis=0)  # S_lamda standard deviation

def correlations(oo,slope=True,sfac=1,ifac=1,mx=1):  
    # the larger the two factors, the more data will be accepted
    plt.figure(5)
    #flt0=np.where((o2a.slope[:,:,1]<sfac*np.abs(o2a.slope[:,:,0])) )
    flt0=np.where((oo.slope[:,:,1]<sfac*np.abs(oo.slope[:,:,0])) & (oo.inter[:,:,1]<ifac*np.abs(oo.inter[:,:,0])))
    print('Accepted ',len(flt0[0]),' out of ',oo.nf * oo.nz,' points.')
    #plt.plot(o2a.sl[flt0[0],flt0[1],l],o2a.slope[flt0[0],flt0[1],0],'k.')
    xx=1e6*oo.co2[flt0[0],flt0[1]]
    if slope:
        yy=oo.slope[flt0[0],flt0[1],0]
    else:
        yy=oo.inter[flt0[0],flt0[1],0]
    flt=np.where((~np.isnan(xx)) & (yy>-mx) & (yy<mx))
    xx=xx[flt]
    yy=yy[flt]
    plt.plot(xx,yy,'k.')
    res=np.polyfit(xx,yy,1)
    x0=np.array([390,400,404])
    y0=res[0]*x0+res[1]
    plt.plot(x0,y0)
    print('correlation, slope')
    print(np.corrcoef(xx,yy)[0,1],res[0])
    #plt.ylim(-1,1)
    plt.xlabel('XCO2')
    if slope: plt.ylabel('Slope')
    if not slope: plt.ylabel('Offset')

def corr_band2band(o1,o2,sfac=1,ifac=1,mx=0.5,slope=True):
    plt.figure(6)
    flt0=np.where((o1.slope[:,:,1]<sfac*np.abs(o1.slope[:,:,0])) & (o1.inter[:,:,1]<ifac*np.abs(o1.inter[:,:,0])) &
                  (o2.slope[:,:,1]<sfac*np.abs(o2.slope[:,:,0])) & (o2.inter[:,:,1]<ifac*np.abs(o2.inter[:,:,0])) )
    if slope:
        xx=o1.slope[flt0[0],flt0[1],0]
        yy=o2.slope[flt0[0],flt0[1],0]
    else:
        xx=o1.inter[flt0[0],flt0[1],0]
        yy=o2.inter[flt0[0],flt0[1],0]
    flt=np.where((np.abs(xx)<mx) & (np.abs(yy)<mx))
    plt.plot(xx[flt],yy[flt],'k.')
    plt.plot([-mx,mx],[-mx,mx],'k--',label='1:1')
    if slope:
        lab='slope'
    else:
        lab='intercept'
    i1=o1.file.rfind('/')+1
    xlab=o1.file[i1:]+' '+lab
    plt.xlabel(xlab)
    i2=o2.file.rfind('/')+1
    ylab=o2.file[i2:]+' '+lab
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(lab+' correlation:'+str(round(np.corrcoef(xx[flt],yy[flt])[0,1],3)))
    
def imagery(png,frame=None,extent=None):
        if frame is None:
            f,frame=plt.subplots()
        img = png[0]
        wesn= png[1]
        img = mpimg.imread(img)
        frame.imshow(img,extent=wesn)
        if not extent is None:
            frame.set_xlim(extent[0],extent[1])
            frame.set_ylim(extent[2],extent[3])
        

def main():                                                           
    #o2a_file  = '/Users/schmidt/rtm/ocosim/dat/new/data_all_20151206_o2a_3879_3981.h5'
    #wco2_file = '/Users/schmidt/rtm/ocosim/dat/new/data_all_20151206_wco2_3879_3981.h5'
    #sco2_file = '/Users/schmidt/rtm/ocosim/dat/new/data_all_20151206_sco2_3879_3981.h5'
    
    #o2a_file  = '/Users/schmidt/rtm/er3toco/oco/data_all_20150622_o2a_2585_2614.h5'
    #wco2_file = '/Users/schmidt/rtm/er3toco/oco/data_all_20150622_wco2_2585_2614.h5'
    #sco2_file = '/Users/schmidt/rtm/er3toco/oco/data_all_20150622_sco2_2585_2614.h5'
    
    o2a_file  = '/Users/schmidt/rtm/ocosim/dat/new/data_all_20150622_o2a_2585_2614.h5'
    wco2_file = '/Users/schmidt/rtm/ocosim/dat/new/data_all_20150622_wco2_2585_2614.h5'
    sco2_file = '/Users/schmidt/rtm/ocosim/dat/new/data_all_20150622_sco2_2585_2614.h5' 
    
    png       = ['/Users/schmidt/papers/OCO/dat/20150622.png',[-64.5552,-50.1085,-14.2336,2.2510]]
    
    z =20 #80
    fp=4  #4
    l=10  # channel
    
    o1   = OCOSIM(o2a_file)      
    o2   = OCOSIM(wco2_file)
    o3   = OCOSIM(sco2_file)
    for i in range(8):
        o1.slopes(i)
        o2.slopes(i)
        o3.slopes(i)
    
    """
    These are the most important plots because they show the slope/intercept
    and their standard deviations – this is what's used downstream
    """
    if False:
        #o3.plot_perturbation(z,fp) # shows single location example fit
        o1.plot_line(fp,l)           # shows single footprint
    
    if False: # visualize radiance along with XCO2 and psur
        o2.visualize_rad(fp=0)
    
    if False:    
        f,x=plt.subplots(1,3,figsize=(10,4))
        s=20
        o1.visualize_radfield(frame=x[0],withobs=True,s=s)
        o2.visualize_radfield(frame=x[1],withobs=True,s=s)
        imagery(png,extent=o1.dom,frame=x[2])
        o2.visualize_co2(frame=x[2])
        #o3.visualize_radfield(frame=x[2],withobs=True,s=s)
        f.tight_layout()
    
    if True:
        correlations(o1,slope=True,sfac=2,ifac=2,mx=1) # correlations between slope/intercept and XCO2
        #corr_band2band(o1,o3,mx=1,slope=False) # band-to-band correlations intercept or slope
    return(o1,o2,o3)

if __name__=='__main__':
    o1,o2,o3=main()
    pass
