import h5py
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import bisect as bs

def create_oco_atm(sat=None, o2mix=0.20935, output='zpt.h5'):
    """
    
    """
    # --------- Constants ------------
    Rd = 287.052874
    EPSILON = 0.622
    kb = 1.380649e-23
    g = 9.81
    # ---------------------------------
    if sat == None:
        sys.exit("[Error] sat information must be provided!")
    elif sat != None:
        # Get reanalysis from met and CO2 prior sounding data
        oco_met = h5py.File(sat.fnames['oco_met'][0], 'r')
        lon_oco_l1b = oco_met['SoundingGeometry/sounding_longitude'][...]
        lat_oco_l1b = oco_met['SoundingGeometry/sounding_latitude'][...]
        logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])

        hprf_l = oco_met['Meteorology/height_profile_met'][...][logic][:, ::-1]
        qprf_l = oco_met['Meteorology/specific_humidity_profile_met'][...][logic][:, ::-1]      # specific humidity mid grid
        sfc_p = oco_met['Meteorology/surface_pressure_met'][...][logic]
        tprf_l = oco_met['Meteorology/temperature_profile_met'][...][logic][:, ::-1]          # temperature mid grid in K
        pprf_l = oco_met['Meteorology/vector_pressure_levels_met'][...][logic][:, ::-1]      # pressure mid grid in Pa
        o3mrprf_l = oco_met['Meteorology/ozone_profile_met'][...][logic][:, ::-1] # kg kg-1
        uprf_l = oco_met['Meteorology/wind_u_profile_met'][...][logic][:, ::-1]
        vprf_l = oco_met['Meteorology/wind_v_profile_met'][...][logic][:, ::-1]
        sfc_gph = oco_met['Meteorology/gph_met'][...][logic]
        oco_met.close()

        oco_co2_aprior = h5py.File(sat.fnames['oco_co2prior'][0], 'r')
        lon_oco_l1b = oco_co2_aprior['SoundingGeometry/sounding_longitude'][...]
        lat_oco_l1b = oco_co2_aprior['SoundingGeometry/sounding_latitude'][...]
        logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
        co2_prf_l = oco_co2_aprior['CO2Prior/co2_prior_profile_cpr'][...][logic][:, ::-1]
        oco_co2_aprior.close()

        oco_lite = h5py.File(sat.fnames['oco_lite'][0], 'r')
        lon_oco_lite = oco_lite['longitude'][...]
        lat_oco_lite = oco_lite['latitude'][...]
        logic = (lon_oco_lite>=sat.extent[0]) & (lon_oco_lite<=sat.extent[1]) & (lat_oco_lite>=sat.extent[2]) & (lat_oco_lite<=sat.extent[3])
        alb_o2a = np.nanmean(oco_lite["Retrieval"]['albedo_o2a'][...][logic])
        alb_wco2 = np.nanmean(oco_lite["Retrieval"]['albedo_wco2'][...][logic])
        alb_sco2 = np.nanmean(oco_lite["Retrieval"]['albedo_sco2'][...][logic])
        sza = np.nanmean(oco_lite["solar_zenith_angle"][...][logic])
        vza = np.nanmean(oco_lite["sensor_zenith_angle"][...][logic])
        oco_lite.close()


        # convert invalid value -999999 to NaN
        for var in [hprf_l, qprf_l, sfc_p, tprf_l, pprf_l, sfc_gph, co2_prf_l]:
            var[var==-999999] = np.nan

        # Ap, Bp from http://wiki.seas.harvard.edu/geos-chem/index.php/GEOS-Chem_vertical_grids#Hybrid_grid
        Ap =  np.array([0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
                        3.257081e+01, 3.898201e+01, 4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
                        7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01, 1.091817e+02, 1.189586e+02,
                        1.286959e+02, 1.429100e+02, 1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
                        2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02, 2.243630e+02, 2.168650e+02,
                        2.011920e+02, 1.769300e+02, 1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
                        7.851231e+01, 6.660341e+01, 5.638791e+01, 4.764391e+01, 4.017541e+01, 3.381001e+01,
                        2.836781e+01, 2.373041e+01, 1.979160e+01, 1.645710e+01, 1.364340e+01, 1.127690e+01,
                        9.292942e+00, 7.619842e+00, 6.216801e+00, 5.046801e+00, 4.076571e+00, 3.276431e+00,
                        2.620211e+00, 2.084970e+00, 1.650790e+00, 1.300510e+00, 1.019440e+00, 7.951341e-01,
                        6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
                        1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
                        1.000000e-02])

        Bp =  np.array([1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
                        8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
                        7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
                        5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
                        2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
                        6.960025e-03, 8.175413e-09, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00])

        P_edge_sfc = np.zeros((sfc_p.shape[0], 73))
        for i in range(73):
            P_edge_sfc[:, i] = sfc_p
        P_edge = Ap*100+Bp*P_edge_sfc
        dP = P_edge[:, 1:] - P_edge[:, :-1]
        log_P_ratio = np.log(P_edge[:, :-1] / P_edge[:, 1:])
        P_mid = (P_edge[:, 1:] + P_edge[:, :-1])/2
        
        r = qprf_l/(1-qprf_l)# mass mixing ratio
        eprf_l = pprf_l*r/(EPSILON+r)
        #Tv = tprf_l/(1-eprf_l/pprf_l*(1-EPSILON))
        Tv = tprf_l/(1-(r/(r+EPSILON))*(1-EPSILON))
        dz_hydrostatic = (Rd*Tv)/g*log_P_ratio

        h_edge = np.empty((sfc_p.shape[0], 73))
        h_edge[:, 0] = sfc_gph
        h_edge[:, 1:] = np.cumsum(dz_hydrostatic[:, :], axis=1) + np.repeat(sfc_gph.reshape(len(sfc_gph), 1), repeats=72, axis=1)
        
        air_layer = pprf_l/(kb*tprf_l)/1e6  # air number density in molec/cm3
        o2_layer = air_layer*o2mix          # O2 number density in molec/cm3
        h2o_layer = eprf_l/(kb*tprf_l)/1e6  # H2O number density in molec/cm3
        co2_layer = air_layer*co2_prf_l     # CO2 number density in molec/cm3
        air_ml = 28.0134*(1-0.20935) + 31.999*0.20935
        o3_layer = air_layer*air_ml*o3mrprf_l/47.9982     # O3 number density in molec/cm3
        h2o_vmr = h2o_layer/air_layer       # H2O volume mixing ratio
        


        pprf_lay_mean   = np.nanmean(pprf_l, axis=0)/100    # pressure mid grid in hPa
        tprf_lay_mean   = np.nanmean(tprf_l, axis=0)         # temperature mid grid in K
        uprf_lay_mean   = np.nanmean(uprf_l, axis=0)    # pressure mid grid in hPa
        vprf_lay_mean   = np.nanmean(vprf_l, axis=0) 
        d_o2_lay_mean   = np.nanmean(o2_layer, axis=0)
        d_co2_lay_mean  = np.nanmean(co2_layer, axis=0)
        d_h2o_lay_mean  = np.nanmean(h2o_layer, axis=0)
        d_o3_lay_mean   = np.nanmean(o3_layer, axis=0)
        dzf             = np.nanmean(dz_hydrostatic, axis=0)/1000      # height diff in km
        hprf_lay_mean   = np.nanmean(hprf_l, axis=0)/1000     # height mid grid in km
        hprf_lev_mean   = np.nanmean(h_edge, axis=0)/1000        # height edge in km
        pprf_lev_mean   = np.nanmean(P_edge, axis=0)/100        # pressure edge in hPa

    # new height edge
    h_edge    = np.concatenate((np.linspace(hprf_lev_mean[0], 5, 11), 
                                np.array([5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])))
    h_lay = np.mean([h_edge[1:], h_edge[:-1]], axis=0)              # height mid grid in km
    dz = h_edge[1:] - h_edge[:-1]

    p_edge, t_edge, _, _, d_o2_lev, d_co2_lev, d_h2o_lev, d_o3_lev = atm_interp(pprf_lay_mean, hprf_lay_mean, tprf_lay_mean, h_edge, uprf_lay_mean, vprf_lay_mean, d_o2_lay_mean, d_co2_lay_mean, d_h2o_lay_mean, d_o3_lay_mean, layer=True)
    p_lay, t_lay, u_lay, v_lay, d_o2_lay, d_co2_lay, d_h2o_lay, d_o3_lay = atm_interp(pprf_lay_mean, hprf_lay_mean, tprf_lay_mean, h_lay, uprf_lay_mean, vprf_lay_mean, d_o2_lay_mean, d_co2_lay_mean, d_h2o_lay_mean, d_o3_lay_mean, layer=True)
    h2o_vmr = d_h2o_lay/(d_o2_lay/0.20935)

    if os.path.isfile(output): 
        print(f'[Warning] Output file {output} exists - overwriting!')
    print('Saving to file '+output)
    h5_output = h5py.File(output, 'w')
    h5_output.create_dataset('h_edge',      data=h_edge)
    h5_output.create_dataset('p_edge',      data=p_edge)
    h5_output.create_dataset('t_edge',      data=t_edge)
    h5_output.create_dataset('h_lay',       data=h_lay)
    h5_output.create_dataset('p_lay',       data=p_lay)
    h5_output.create_dataset('t_lay',       data=t_lay)
    h5_output.create_dataset('u_lay',       data=u_lay)
    h5_output.create_dataset('v_lay',       data=v_lay)
    h5_output.create_dataset('d_o2_lay',    data=d_o2_lay)
    h5_output.create_dataset('d_co2_lay',   data=d_co2_lay)
    h5_output.create_dataset('d_h2o_lay',   data=d_h2o_lay)
    h5_output.create_dataset('d_o3_lay',    data=d_o3_lay)
    h5_output.create_dataset('d_o2_lev',    data=d_o2_lev)
    h5_output.create_dataset('d_co2_lev',   data=d_co2_lev)
    h5_output.create_dataset('d_h2o_lev',   data=d_h2o_lev)
    h5_output.create_dataset('d_o3_lev',    data=d_o3_lev)
    h5_output.create_dataset('h2o_vmr',     data=h2o_vmr)
    h5_output.create_dataset('dz',          data=dz)
    h5_output.create_dataset('albedo_o2a',  data=alb_o2a)
    h5_output.create_dataset('albedo_wco2', data=alb_wco2)
    h5_output.create_dataset('albedo_sco2', data=alb_sco2)
    h5_output.create_dataset('sza',         data=sza)
    h5_output.create_dataset('vza',         data=vza)

    return None


def atm_interp(pressure, altitude, temperature, 
                altitude_to_interp,
                u=None, v=None,
                d_o2=None, d_co2=None, d_h2o=None, d_o3=None,
                layer=False):

    """
    Use Barometric formula (https://en.wikipedia.org/wiki/Barometric_formula)
    to interpolate pressure from height and temperature

    Input:
        pressure: numpy array, original pressure in hPa
        altitude: numpy array, original altitude in km
        temperature: numpy array, original temperature in K
        altitude_to_interp: numpy array, altitude to be interpolate
        temperature_interp: numpy array, temperature to be interpolate

    Output:
        pn: interpolated pressure based on the input
    """

    indices = np.argsort(altitude)
    h       = np.float_(altitude[indices])
    p       = np.float_(pressure[indices])
    t       = np.float_(temperature[indices])

    indices = np.argsort(altitude_to_interp)
    #hn      = np.float_(altitude_to_interp[indices])
    hn      = altitude_to_interp
    # linear interpolation for temperature
    tn      = np.interp(hn, altitude, temperature)

    n = p.size - 1
    a = 0.5*(t[1:]+t[:-1]) / (h[:-1]-h[1:]) * np.log(p[1:]/p[:-1])
    z = 0.5*(h[1:]+h[:-1])

    z0  = np.min(z) ; z1  = np.max(z)
    hn0 = np.min(hn); hn1 = np.max(hn)

    if hn0 < z0:
        a = np.hstack((a[0], a))
        z = np.hstack((hn0, z))
        if z0 - hn0 > 2.0:
            print('Warning [atm_interp_pressure]: Standard atmosphere not sufficient (lower boundary).')

    if hn1 > z1:
        a = np.hstack((a, z[n-1]))
        z = np.hstack((z, hn1))
        if hn1-z1 > 10.0:
            print('Warning [atm_interp_pressure]: Standard atmosphere not sufficient (upper boundary).')

    an = np.interp(hn, z, a)
    pn = np.zeros_like(hn)

    if hn.size == 1:
        hi = np.argmin(np.abs(hn-h))
        pn = p[hi]*np.exp(-an*(hn-h[hi])/tn)
        return pn

    for i in range(pn.size):
        hi = np.argmin(np.abs(hn[i]-h))
        pn[i] = p[hi]*np.exp(-an[i]*(hn[i]-h[hi])/tn[i])

    dp = pn[:-1] - pn[1:]
    pl = 0.5 * (pn[1:]+pn[:-1])
    zl = 0.5 * (hn[1:]+hn[:-1])

    for i in range(n-2):
        indices = (zl >= h[i]) & (zl < h[i+1])
        ind = np.where(indices==True)[0]
        ni  = indices.sum()
        if ni >= 2:
            dpm = dp[ind].sum()

            i0 = np.min(ind)
            i1 = np.max(ind)

            x1 = pl[i0]
            x2 = pl[i1]
            y1 = dp[i0]
            y2 = dp[i1]

            bb = (y2-y1) / (x2-x1)
            aa = y1 - bb*x1
            rescale = dpm / (aa+bb*pl[indices]).sum()

            if np.abs(rescale-1.0) > 0.1:
                print('------------------------------------------------------------------------------')
                print('Warning [atm_interp_pressure]:')
                print('Warning: pressure smoothing failed at ', h[i], '...', h[i+1])
                print('rescale=', rescale)
                print('------------------------------------------------------------------------------')
            else:
                dp[indices] = rescale*(aa+bb*pl[indices])

    for i in range(dp.size):
        pn[i+1] = pn[i] - dp[i] 
    
    if layer==False:
        return pn, tn
    
    elif layer==True:
        if 1:
            u_lay, v_lay = np.zeros_like(hn), np.zeros_like(hn)
            d_o2_lay, d_co2_lay, d_h2o_lay, d_o3_lay = np.zeros_like(hn), np.zeros_like(hn), np.zeros_like(hn), np.zeros_like(hn)
            for i in range(len(altitude_to_interp)):
                lev_index = bs.bisect(h, hn[i])-1
                d_o2_lay[i] = pn[i]/p[lev_index]*d_o2[lev_index]
                d_co2_lay[i] = pn[i]/p[lev_index]*d_co2[lev_index]
                d_h2o_lay[i] = pn[i]/p[lev_index]*d_h2o[lev_index]
                d_o3_lay[i] = pn[i]/p[lev_index]*d_o3[lev_index]
                h_ratio = (hn[i]-h[lev_index])/(h[lev_index+1]-h[lev_index])
                u_lay[i] = u[lev_index]*(1-h_ratio)+u[lev_index+1]*(h_ratio)
                v_lay[i] = v[lev_index]*(1-h_ratio)+v[lev_index+1]*(h_ratio)
            
            return pn, tn, u_lay, v_lay, d_o2_lay, d_co2_lay, d_h2o_lay, d_o3_lay
        #except:
        #    sys.exit("[Error] Must provide u, v, O2, CO2, H2O number densities!")
        