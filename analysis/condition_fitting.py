import sys
sys.path.append('/Users/yuch8913/miniforge3/envs/er3t_env/lib/python3.8/site-packages')
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager, cm, colors
import numpy as np
import copy
from oco_post_class_ywc import OCOSIM
from scipy import interpolate
from scipy.ndimage import uniform_filter
from  scipy.optimize import curve_fit
from scipy import linalg
import geopy.distance
import seaborn as sns
from tool_code import *
import os, pickle 
from haversine import Unit, haversine_vector
import uncertainties.unumpy as unp
import uncertainties as unc
from util.oco_cfg import grab_cfg, output_h5_info, nan_array, ax_lon_lat_label

font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

def exp_decay_const_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def main(cfg_csv='20181018_central_asia_2_test4.csv'):
    # '20181018_central_asia_2_test4.csv'
    # '20150622_amazon.csv'
    # '20181018_central_asia_2_test6.csv'
    # 20190621_australia_2.csv

    cfg_dir = '../simulation/cfg'
    cfg_info = grab_cfg(f'{cfg_dir}/{cfg_csv}')
    if 'o2' in cfg_info.keys():
        id_num = output_h5_info(f'{cfg_dir}/{cfg_csv}', 'o2')[22:31]
    else:
        raise IOError('No output files are recorded!')
    
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:]))    # day
                              
    cfg_name = cfg_info['cfg_name']
    case_name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    extent_png = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]

    if not os.path.exists('output'):
        os.makedirs('output')
    # img_dir = f'output/{case_name_tag}'
    img_dir = f'output/{case_name_tag}_condition'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    
    df = pd.read_csv(f'{img_dir}/{cfg_name}_fitting_result_condition.txt', sep=',')
    # head = 'alb,sza,aod,'
    # for i in range(3):
    #     head += 'slope_%s_amp,slope_%s_dec,inter_%s_amp,inter_%s_dec,' %(channel_list[i], channel_list[i], channel_list[i], channel_list[i])
    #     head += 'slope_%s_amp_unc,slope_%s_dec_unc,inter_%s_amp_unc,inter_%s_dec_unc,' %(channel_list[i], channel_list[i], channel_list[i], channel_list[i])
    # if not os.path.isfile(f'{img_dir}/{cfg_name}_fitting_conbination.txt'):
    #     with open(f'{img_dir}/{cfg_name}_fitting_conbination.txt', 'w') as f:
    #         head = 'band, sza, aod, slope_amp_a, slope_amp_b, slope_amp_c, slope_dec_a, slope_dec_b, slope_dec_c, inter_amp_a, inter_amp_b, inter_amp_c, inter_dec_a, inter_dec_b, inter_dec_c,'
    #         f.write(head[:-1]+'\n')
    for col in df.columns:
        df[col] = df[col].astype(float)
        
    for band in ['o2a', 'wco2', 'sco2']:
        fig, ((ax11, ax12, ax13, ax14), 
            (ax21, ax22, ax23, ax24),
            (ax31, ax32, ax33, ax34)) = plt.subplots(3, 4, figsize=(24, 12), sharex=False)
        label_size = 16
        tick_size = 13

        ax_list = np.array([(ax11, ax12, ax13, ax14), 
                            (ax21, ax22, ax23, ax24),
                            (ax31, ax32, ax33, ax34)])
        # alb_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # sza_list = np.array([15, 30, 45, 60, 75])
        alb_list = np.array([0.2, 0.3, 0.4, 0.5])
        sza_list = np.array([30, 45, 60])
        aod_list = [0, 0.05, 0.1]
        color_list = ['r', 'g', 'b', 'c', 'm',]
        slope_amp_all, slope_amp_err_all = [], []
        slope_dec_all, slope_dec_err_all = [], []
        inter_amp_all, inter_amp_err_all = [], []
        inter_dec_all, inter_dec_err_all = [], []
        alb_all, sza_all, aod_all = [], [], []
        for i in range(3):
            
            
            aod = aod_list[i]
            df_select = df[df['aod']==aod]
            ax1, ax2, ax3, ax4 = ax_list[i]
            for j in range(len(sza_list)):
                sza = sza_list[j]
                sza_select = df_select[df_select['sza']==sza]
                slope_amp, slope_amp_err = [], []
                slope_dec, slope_dec_err = [], []
                inter_amp, inter_amp_err = [], []
                inter_dec, inter_dec_err = [], []
                for alb in alb_list:
                    alb_select = sza_select[sza_select['alb']==alb]
                    slope_amp.append(alb_select['slope_%s_amp' %band].values[0])
                    slope_amp_err.append(alb_select['slope_%s_amp_unc' %band].values[0])
                    slope_dec.append(alb_select['slope_%s_dec' %band].values[0])
                    slope_dec_err.append(alb_select['slope_%s_dec_unc' %band].values[0])
                    inter_amp.append(alb_select['inter_%s_amp' %band].values[0])
                    inter_amp_err.append(alb_select['inter_%s_amp_unc' %band].values[0])
                    inter_dec.append(alb_select['inter_%s_dec' %band].values[0])
                    inter_dec_err.append(alb_select['inter_%s_dec_unc' %band].values[0])
                slope_amp = np.array(slope_amp)
                slope_amp_err = np.array(slope_amp_err)
                slope_dec = np.array(slope_dec)
                slope_dec_err = np.array(slope_dec_err)
                inter_amp = np.array(inter_amp) 
                inter_amp_err = np.array(inter_amp_err)
                inter_dec = np.array(inter_dec)
                inter_dec_err = np.array(inter_dec_err)
                
                   
                            
                
                for data_list in [slope_amp, inter_amp, slope_amp_err, inter_amp_err]:
                    data_list[data_list>5] = np.nan
                    data_list[data_list==0] = np.nan
                for data_list in [slope_dec, inter_dec, slope_dec_err, inter_dec_err]:
                    data_list[data_list==0] = np.nan
                slope_dec = 1/slope_dec
                slope_dec_err = 1/slope_dec_err
                inter_dec = 1/inter_dec
                inter_dec_err = 1/inter_dec_err
                for data_list in [slope_dec, inter_dec, slope_dec_err, inter_dec_err]:
                    data_list[data_list>100] = np.nan
                    data_list[data_list<1] = np.nan
                
                slope_amp_all.extend(slope_amp)
                slope_amp_err_all.extend(slope_amp_err)
                slope_dec_all.extend(slope_dec)
                slope_dec_err_all.extend(slope_dec_err)
                inter_amp_all.extend(inter_amp)
                inter_amp_err_all.extend(inter_amp_err)
                inter_dec_all.extend(inter_dec)
                inter_dec_err_all.extend(inter_dec_err)
                alb_all.extend(alb_list)
                sza_all.extend([sza_list[j]]*len(slope_amp))
                aod_all.extend([aod]*len(slope_amp)) 
                
                for data_list in [slope_amp_all, slope_amp_err_all, slope_dec_all, slope_dec_err_all,
                                  inter_amp_all, inter_amp_err_all, inter_dec_all, inter_dec_err_all,
                                  alb_all, sza_all, aod_all]:
                    data_list = np.array(data_list)
                
                write_data = f'{band},{sza},{aod},'
                ax1.errorbar(alb_list, slope_amp, #yerr=slope_amp_err, 
                             fmt='o', color=color_list[j], label=sza)
                # popt, pcov = curve_fit(exp_decay_const_func, alb_list, slope_amp, #bounds=([1e-5, 1e-5], [15, 15,]),
                #                        #p0=(0.1, 0.7),
                #                        maxfev=50000, sigma=slope_amp_err, 
                #                        absolute_sigma=True,)
                # write_data += f'{popt[0]}, {popt[1]}, {popt[2]},'
                # ax1.plot(alb_list, exp_decay_const_func(alb_list, *popt), color=color_list[i], linestyle='--')
                ax2.errorbar(alb_list, slope_dec, #yerr=slope_dec_err, 
                             fmt='o', color=color_list[j], label=sza)
                # popt, pcov = curve_fit(exp_decay_const_func, alb_list, slope_dec, #bounds=([1e-5, 1e-5], [15, 15,]),
                #                        #p0=(0.1, 0.7),
                #                        maxfev=50000, sigma=slope_dec_err, 
                #                        absolute_sigma=True,)
                # write_data += f'{popt[0]}, {popt[1]}, {popt[2]},'
                # ax2.plot(alb_list, exp_decay_const_func(alb_list, *popt), color=color_list[i], linestyle='--')
                ax3.errorbar(alb_list, inter_amp, #yerr=inter_amp_err, 
                             fmt='o', color=color_list[j], label=sza)
                # popt, pcov = curve_fit(exp_decay_const_func, alb_list, inter_amp, #bounds=([1e-5, 1e-5], [15, 15,]),
                #                        #p0=(0.1, 0.7),
                #                        maxfev=50000, sigma=inter_amp_err, 
                #                        absolute_sigma=True,)
                # write_data += f'{popt[0]}, {popt[1]}, {popt[2]},'
                # ax3.plot(alb_list, exp_decay_const_func(alb_list, *popt), color=color_list[i], linestyle='--')
                ax4.errorbar(alb_list, inter_dec, #yerr=inter_dec_err, 
                             fmt='o', color=color_list[j], label=sza)
                # popt, pcov = curve_fit(exp_decay_const_func, alb_list, inter_dec, #bounds=([1e-5, 1e-5], [15, 15,]),
                #                        #p0=(0.1, 0.7),
                #                        maxfev=50000, sigma=inter_dec_err, 
                #                        absolute_sigma=True,)
                # write_data += f'{popt[0]}, {popt[1]}, {popt[2]},'
                # ax4.plot(alb_list, exp_decay_const_func(alb_list, *popt), color=color_list[i], linestyle='--')
                with open(f'{img_dir}/{cfg_name}_fitting_conbination.txt', 'a') as f:
                    f.write(write_data[:-1]+'\n')
            ax1.set_title('Slope amplitude', fontsize=label_size)
            ax2.set_title('Slope e-folding_distance', fontsize=label_size)
            ax3.set_title('Intercept amplitude', fontsize=label_size)
            ax4.set_title('Intercept e-folding_distance', fontsize=label_size)
            ax1.legend(title='SZA', fontsize=label_size)
            ax2.legend(title='SZA', fontsize=label_size)
            ax3.legend(title='SZA', fontsize=label_size)
            ax4.legend(title='SZA', fontsize=label_size)

        label_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
        for i, ax in enumerate(ax_list.flatten()):
            label_text = f'({label_list[i]})'
            ax.set_xlabel('surface albedo', fontsize=label_size)
            ax.tick_params(axis='both', labelsize=tick_size)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.hlines(0, 0, xmax, linestyle='--', color='grey')
            ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, fontsize=label_size, color='k')
        
        save_figure(fig, f'fit_{band}_band.png', img_dir,)# pad=5.0)
        plt.close()
        plt.clf()
        final = np.array([alb_all, sza_all, aod_all, 
                      slope_amp_all, slope_dec_all, 
                      inter_amp_all, inter_dec_all]).T
        print('test:', final.shape)
        df_final = pd.DataFrame(final,
                                columns=['alb', 'sza', 'aod', 'slope_amp', 'slope_dec', 'inter_amp', 'inter_dec'],
                                )   
        df_final.to_csv(f'{img_dir}/{cfg_name}_{band}_fitting_conbination_all.csv', index=False)

        
    for band in ['o2a', 'wco2', 'sco2']:
        fig, (ax11, ax12, ax13, ax14) = plt.subplots(1, 4, figsize=(24, 4), sharex=False)
        aod = aod_list[0]
        df_select = df[df['aod']==aod]
        ax1, ax2, ax3, ax4 = (ax11, ax12, ax13, ax14)
        slope_amp_all, slope_amp_err_all = [], []
        slope_dec_all, slope_dec_err_all = [], []
        inter_amp_all, inter_amp_err_all = [], []
        inter_dec_all, inter_dec_err_all = [], []
        alb_all, sza_all, aod_all = [], [], []
        
        for j in range(len(sza_list)):
            sza = sza_list[j]
            sza_select = df_select[df_select['sza']==sza]
            slope_amp, slope_amp_err = [], []
            slope_dec, slope_dec_err = [], []
            inter_amp, inter_amp_err = [], []
            inter_dec, inter_dec_err = [], []
            for alb in alb_list:
                alb_select = sza_select[sza_select['alb']==alb]
                slope_amp.append(alb_select['slope_%s_amp' %band].values[0])
                slope_amp_err.append(alb_select['slope_%s_amp_unc' %band].values[0])
                slope_dec.append(alb_select['slope_%s_dec' %band].values[0])
                slope_dec_err.append(alb_select['slope_%s_dec_unc' %band].values[0])
                inter_amp.append(alb_select['inter_%s_amp' %band].values[0])
                inter_amp_err.append(alb_select['inter_%s_amp_unc' %band].values[0])
                inter_dec.append(alb_select['inter_%s_dec' %band].values[0])
                inter_dec_err.append(alb_select['inter_%s_dec_unc' %band].values[0])
            slope_amp = np.array(slope_amp)
            slope_amp_err = np.array(slope_amp_err)
            slope_dec = np.array(slope_dec)
            slope_dec_err = np.array(slope_dec_err)
            inter_amp = np.array(inter_amp) 
            inter_amp_err = np.array(inter_amp_err)
            inter_dec = np.array(inter_dec)
            inter_dec_err = np.array(inter_dec_err)
            
                
                        
            
            for data_list in [slope_amp, inter_amp, slope_amp_err, inter_amp_err]:
                data_list[data_list>5] = np.nan
                data_list[data_list==0] = np.nan
            for data_list in [slope_dec, inter_dec, slope_dec_err, inter_dec_err]:
                data_list[data_list==0] = np.nan
            slope_dec = 1/slope_dec
            slope_dec_err = 1/slope_dec_err
            inter_dec = 1/inter_dec
            inter_dec_err = 1/inter_dec_err
            for data_list in [slope_dec, inter_dec, slope_dec_err, inter_dec_err]:
                data_list[data_list>30] = np.nan
                data_list[data_list<1] = np.nan
            
            slope_amp_all.extend(slope_amp)
            slope_amp_err_all.extend(slope_amp_err)
            slope_dec_all.extend(slope_dec)
            slope_dec_err_all.extend(slope_dec_err)
            inter_amp_all.extend(inter_amp)
            inter_amp_err_all.extend(inter_amp_err)
            inter_dec_all.extend(inter_dec)
            inter_dec_err_all.extend(inter_dec_err)
            alb_all.extend(alb_list)
            sza_all.extend([sza_list[j]]*len(slope_amp))
            aod_all.extend([aod]*len(slope_amp)) 
        
        if band == 'o2a':
            slope_amp_all.append(0.263)
            slope_amp_err_all.append(0.088)
            slope_dec_all.append(4.86)
            slope_dec_err_all.append(0.99)
            inter_amp_all.append(0.667)
            inter_amp_err_all.append(0.257)
            inter_dec_all.append(2.81)
            inter_dec_err_all.append(0.33)
            alb_all.append(0.288)
            sza_all.append(48.5)
            aod_all.append(0)
        elif band == 'wco2':
            slope_amp_all.append(0.120)
            slope_amp_err_all.append(0.033)
            slope_dec_all.append(5.06)
            slope_dec_err_all.append(0.86)
            inter_amp_all.append(0.667)
            inter_amp_err_all.append(0.213)
            inter_dec_all.append(3.03)
            inter_dec_err_all.append(0.31)
            alb_all.append(0.375)
            sza_all.append(48.5)
            aod_all.append(0)
        elif band == 'sco2':
            slope_amp_all.append(0.102)
            slope_amp_err_all.append(0.022)
            slope_dec_all.append(6.17)
            slope_dec_err_all.append(1.04)
            inter_amp_all.append(0.745)
            inter_amp_err_all.append(0.259)
            inter_dec_all.append(2.67)
            inter_dec_err_all.append(0.29)
            alb_all.append(0.370)
            sza_all.append(48.5)
            aod_all.append(0)
        
        # for data_list in [slope_amp_all, slope_amp_err_all, slope_dec_all, slope_dec_err_all,
        #                     inter_amp_all, inter_amp_err_all, inter_dec_all, inter_dec_err_all,
        #                     alb_all, sza_all, aod_all]:
        #     print(np.array(data_list).shape)
        sza_all = np.array(sza_all)
        sza_all = np.cos(np.deg2rad(sza_all))
        final = np.array([alb_all, sza_all, aod_all, 
                      slope_amp_all, slope_dec_all, 
                      inter_amp_all, inter_dec_all]).T
        print('test:', final.shape)
        df_final = pd.DataFrame(final,
                                columns=['alb', 'sza', 'aod', 'slope_amp', 'slope_dec', 'inter_amp', 'inter_dec'],
                                )
        df_final.dropna(inplace=True)
        alb_all = df_final['alb'].values
        sza_all = df_final['sza'].values
        slope_amp_all = df_final['slope_amp'].values
        slope_dec_all = df_final['slope_dec'].values
        inter_amp_all = df_final['inter_amp'].values
        inter_dec_all = df_final['inter_dec'].values
        
        c1 = ax1.scatter(alb_all, sza_all, c=slope_amp_all, cmap='jet')
        cb1 = fig.colorbar(c1, ax=ax1)
        cb1.set_label('$\mathrm{a_s}$', fontsize=label_size)
        c2 = ax2.scatter(alb_all, sza_all, c=slope_dec_all, cmap='jet')
        cb2 = fig.colorbar(c2, ax=ax2)
        cb2.set_label('$\mathrm{d_s}$', fontsize=label_size)
        c3 = ax3.scatter(alb_all, sza_all, c=inter_amp_all, cmap='jet')
        cb3 = fig.colorbar(c3, ax=ax3)
        cb3.set_label('$\mathrm{a_i}$', fontsize=label_size)
        c4 = ax4.scatter(alb_all, sza_all, c=inter_dec_all, cmap='jet')
        cb4 = fig.colorbar(c4, ax=ax4)
        cb4.set_label('$\mathrm{d_i}$', fontsize=label_size)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('surface albedo', fontsize=label_size)
            ax.set_ylabel('sza', fontsize=label_size)
            ax.tick_params(axis='both', labelsize=tick_size)
        fig.tight_layout()
        plt.show()
        
        # print(slope_amp_all)
        # print(slope_dec_all)
        # print(inter_amp_all)
        # print(inter_dec_all)
        
        fig = plt.figure(figsize=(24, 4))
        ax1=fig.add_subplot(1,4,1, projection='3d')
        ax2=fig.add_subplot(1,4,2, projection='3d')
        ax3=fig.add_subplot(1,4,3, projection='3d')
        ax4=fig.add_subplot(1,4,4, projection='3d')
        cos_sza = np.cos(np.deg2rad(sza_list))
        xx, yy = np.meshgrid(alb_list, cos_sza)
        
        # ax1
        ax1.scatter3D(alb_all, sza_all, slope_amp_all)
        fittedParameters, pcov = curve_fit(two_var_func, [alb_all, sza_all], slope_amp_all,
                                                          #p0 = initialParameters
                                                          )
        Z = two_var_func(np.array([xx, yy]), *fittedParameters)
        ax1.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)
        print('band:', band, 'slope_amp:', fittedParameters)
        ax1.set_zlabel('$\mathrm{a_s}$', fontsize=label_size)
        # ax2
        ax2.scatter3D(alb_all, sza_all, slope_dec_all)
        fittedParameters, pcov = curve_fit(two_var_func, [alb_all, sza_all], slope_dec_all,
                                                          #p0 = initialParameters
                                                          )
        Z = two_var_func(np.array([xx, yy]), *fittedParameters)
        ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)
        print('band:', band, 'slope_dec:', fittedParameters)
        ax2.set_zlabel('$\mathrm{d_s}$', fontsize=label_size)
        # ax3
        ax3.scatter3D(alb_all, sza_all, inter_amp_all)
        fittedParameters, pcov = curve_fit(two_var_func, [alb_all, sza_all], inter_amp_all,
                                                          #p0 = initialParameters
                                                          )
        Z = two_var_func(np.array([xx, yy]), *fittedParameters)
        ax3.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)
        print('band:', band, 'inter_amp:', fittedParameters)
        ax3.set_zlabel('$\mathrm{a_i}$', fontsize=label_size)
        # ax4
        ax4.scatter3D(alb_all, sza_all, inter_dec_all)
        fittedParameters, pcov = curve_fit(two_var_func, [alb_all, sza_all], inter_dec_all,
                                                          #p0 = initialParameters
                                                          )
        Z = two_var_func(np.array([xx, yy]), *fittedParameters)
        ax4.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)
        print('band:', band, 'inter_dec:', fittedParameters)
        ax4.set_zlabel('$\mathrm{d_i}$', fontsize=label_size)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('surface albedo', fontsize=label_size)
            ax.set_ylabel('sza', fontsize=label_size)
            ax.tick_params(axis='both', labelsize=tick_size)
        plt.show()
        plt.close()
        
        
            
    
    sys.exit()
    
    # slope_a, slope_b, inter_a, inter_b
    o2a_inter = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[0][2], parameters_cld_distance_list[0][3])
    o2a_slope = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[0][0], parameters_cld_distance_list[0][1])
    wco2_inter = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[1][2], parameters_cld_distance_list[1][3])
    wco2_slope = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[1][0], parameters_cld_distance_list[1][1])
    sco2_inter = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[2][2], parameters_cld_distance_list[2][3])
    sco2_slope = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[2][0], parameters_cld_distance_list[2][1])

    output_csv = pd.DataFrame({'SND': snd[xco2_valid][mask_fp].flatten(),
                                'LON': o1.lon[xco2_valid][mask_fp].flatten(),
                                'LAT': o1.lat[xco2_valid][mask_fp].flatten(),
                                'L2XCO2[ppm]': xco2[xco2_valid][mask_fp].flatten()*1e6,
                                'L2PSUR[kPa]': psur[xco2_valid][mask_fp].flatten()/1000,
                                'i1': o2a_inter,
                                's1': o2a_slope,
                                'i2': wco2_inter,
                                's2': wco2_slope,
                                'i3': sco2_inter,
                                's3': sco2_slope,
                                'weighted_cld_distance': oco_footprint_cld_distance.flatten(),
                                },)
    output_csv['SND'] = output_csv['SND'].apply(lambda x: f'SND{x:.0f}')
    # output_csv.to_csv(f'{cfg_name}_footprint_cld_distance.csv', index=False)

    output_csv.to_csv(f'{cfg_name}_footprint_idealized.csv', index=False)
    # sys.exit()

    inter_lon_lat = (np.array(weighted_cld_data['lon']).reshape(o1.lon2d.shape)[:, 0], 
                     np.array(weighted_cld_data['lat']).reshape(o1.lon2d.shape)[0, :])
    o2a_slope_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                              getattr(o1, 'slope_5avg')[:,:,0], method='nearest')
    o2a_inter_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                              getattr(o1, 'inter_5avg')[:,:,0], method='nearest')
    wco2_slope_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                               getattr(o2, 'slope_5avg')[:,:,0], method='nearest')
    wco2_inter_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                               getattr(o2, 'inter_5avg')[:,:,0], method='nearest')
    sco2_slope_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                               getattr(o3, 'slope_5avg')[:,:,0], method='nearest')
    sco2_inter_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                               getattr(o3, 'inter_5avg')[:,:,0], method='nearest')
    oco_footprint_o2a_slope = o2a_slope_5avg_func(points_footprint)
    oco_footprint_o2a_inter = o2a_inter_5avg_func(points_footprint)
    wco2_footprint_o2a_slope = wco2_slope_5avg_func(points_footprint)
    wco2_footprint_o2a_inter = wco2_inter_5avg_func(points_footprint)
    sco2_footprint_o2a_slope = sco2_slope_5avg_func(points_footprint)
    sco2_footprint_o2a_inter = sco2_inter_5avg_func(points_footprint)

    pxl_by_pxl_output_csv = pd.DataFrame({'SND': snd[xco2_valid][mask_fp].flatten(),
                                        'LON': o1.lon[xco2_valid][mask_fp].flatten(),
                                        'LAT': o1.lat[xco2_valid][mask_fp].flatten(),
                                        'L2XCO2[ppm]': xco2[xco2_valid][mask_fp].flatten()*1e6,
                                        'L2PSUR[kPa]': psur[xco2_valid][mask_fp].flatten()/1000,
                                        'i1': oco_footprint_o2a_inter,
                                        's1': oco_footprint_o2a_slope,
                                        'i2': wco2_footprint_o2a_inter,
                                        's2': wco2_footprint_o2a_slope,
                                        'i3': sco2_footprint_o2a_inter,
                                        's3': sco2_footprint_o2a_slope,
                                        'weighted_cld_distance': oco_footprint_cld_distance.flatten(),
                                        },)
    pxl_by_pxl_output_csv['SND'] = pxl_by_pxl_output_csv['SND'].apply(lambda x: f'SND{x:.0f}')
    pxl_by_pxl_output_csv.to_csv(f'{cfg_name}_footprint_pixel_by_pixel.csv', index=False)
    
    cld_lon, cld_lat, cld_location = cld_position_func(cfg_name)
    
    with h5py.File(f'../simulation/data/{case_name_tag}/pre-data.h5', 'r') as f:
        lon_2d = f['lon'][...]
        lat_2d = f['lat'][...]
        sfh_2d = f['mod/geo/sfh'][...]
        cth0 = f['mod/cld/logic_cld'][...]
    
    extent = [float(loc) for loc in cfg_info['subdomain']]
    mask = np.logical_and(np.logical_and(lon_2d >= extent[0], lon_2d <= extent[1]),
                          np.logical_and(lat_2d >= extent[2], lat_2d <= extent[3]))

    img_file = f'../simulation/data/{case_name_tag}/{cfg_info["png"]}'
    wesn = extent_png
    img = mpimg.imread(img_file)
    lon_dom = extent_analysis[:2]
    lat_dom = extent_analysis[2:]

    
    sfc_alt_plt(img, wesn, lon_dom, lat_dom, 
                lon_2d, lat_2d, sfh_2d, img_dir=img_dir)
    
    cld_dist_plot(o1, img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, cld_dist, img_dir=img_dir)
    
    weighted_cld_dist_plot(o1, img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, weighted_cld_dist, img_dir=img_dir)
    
    slope_intercept_compare_plot(o1, 'O_2-A', 'o2a', pxl_by_pxl_output_csv,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare, img_dir=img_dir)
    
    slope_intercept_compare_plot(o3, 'SCO_2', 'sco2', pxl_by_pxl_output_csv,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare, img_dir=img_dir)
    
    o2a_conti_plot(o1, rad_c3d_compare,
                   img, wesn, lon_dom, lat_dom, label_size=14, img_dir=img_dir)
    
    continuum_fp_compare_plot(o1, o2, o3,
                              img, wesn, lon_dom, lat_dom, 
                              lon_2d, lat_2d, cth0, img_dir=img_dir)
    
    o2a_wvl_select_slope_derivation(cfg_info, o1, img_dir=img_dir)

def two_var_func(data, a1, a2, b):
    x = data[0]
    y = data[1]
    return a1*x+a2*y+b


def setup_plot(wesn, lon_dom, lat_dom, label_size, tick_size=12, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(wesn[0], wesn[1])
    ax.set_ylim(wesn[2], wesn[3])
    ax.vlines(lon_dom, ymin=lat_dom[0], ymax=lat_dom[1], color='k', linewidth=1)
    ax.hlines(lat_dom, xmin=lon_dom[0], xmax=lon_dom[1], color='k', linewidth=1)
    ax_lon_lat_label(ax, label_size=label_size, tick_size=tick_size)
    return fig, ax

def save_figure(fig, file_name, img_dir, pad=None, dpi=300):
    if pad is None:
        fig.tight_layout()
    else:
        fig.tight_layout(pad=pad)
    fig.savefig(f'{img_dir}/{file_name}', dpi=dpi)


def sfc_alt_plt(img, wesn, lon_dom, lat_dom, 
                lon_2d, lat_2d, sfh_2d, label_size=14, img_dir='.'):
    fig, ax = setup_plot(wesn, lon_dom, lat_dom, label_size)
    ax.imshow(img, extent=wesn)
    c = ax.contourf(lon_2d,lat_2d, sfh_2d*1000,
                    cmap='terrain', levels=201, vmin=0, vmax=2000)
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('Surface altitude (m)', fontsize=label_size)
    save_figure(fig, 'surface_altitude.png', img_dir)


def cld_dist_plot(o1, img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, cld_dist, label_size=14, img_dir='.'):
    fig, ax = setup_plot(wesn, lon_dom, lat_dom, label_size)
    ax.imshow(img, extent=wesn)
    c = ax.scatter(o1.lon2d, o1.lat2d, 
                   c=cld_dist, s=5,
                   cmap='Reds', vmin=0, vmax=20)
    ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='b')
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('Cloud distance (km)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    save_figure(fig, 'cloud_distance.png', img_dir)

def weighted_cld_dist_plot(o1, img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, weighted_cld_dist, label_size=14, img_dir='.'):
    fig, ax = setup_plot(wesn, lon_dom, lat_dom, label_size)
    ax.imshow(img, extent=wesn)
    c = ax.scatter(o1.lon2d, o1.lat2d, 
                   c=weighted_cld_dist, s=5,
                   cmap='Reds', vmin=0, vmax=20)
    ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='b')
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('$\mathrm{D_e}$ (km)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    save_figure(fig, 'weighted_cloud_distance.png', img_dir)

def o2a_conti_plot(o1, rad_c3d_compare,
                   img, wesn, lon_dom, lat_dom, label_size=14, img_dir='.'):
    fig, ax = setup_plot(wesn, lon_dom, lat_dom, label_size)
    ax.imshow(img, extent=wesn)
    mask = np.isnan(getattr(o1, rad_c3d_compare)[:,:,-1])
    c = ax.scatter(o1.lon2d, o1.lat2d, 
                   c=getattr(o1, rad_c3d_compare)[:,:,-1], s=5, cmap='Reds')
    ax.scatter(o1.lon2d[mask], o1.lat2d[mask], c='grey', s=5, cmap='Reds')
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('$\mathrm{O_2-A}$ continuum (mW m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    save_figure(fig, f'o2a_conti_{rad_c3d_compare}.png', img_dir)

def slope_intercept_compare_plot(OCO_class, label_tag, file_tag, pxl_by_pxl_output_csv,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare, label_size=14, img_dir='.'):
    
    f, (ax1, ax2) =plt.subplots(1, 2, figsize=(12, 6.75))
    for ax in [ax1, ax2]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax_lon_lat_label(ax, label_size=14, tick_size=12)
    
    mask = ~(cth0>0)
    c1 = ax1.scatter(OCO_class.lon2d[mask], OCO_class.lat2d[mask], 
                     c=getattr(OCO_class, slope_compare)[:,:,0][mask], s=10,
                     cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    idx = 91
    print(pxl_by_pxl_output_csv['SND'][idx])
    print('lon ind:', np.argmin(np.abs(pxl_by_pxl_output_csv['LON'][idx]-OCO_class.lon2d[:, 0])))
    print('lat ind:', np.argmin(np.abs(pxl_by_pxl_output_csv['LAT'][idx]-OCO_class.lat2d[0, :])))
    ax1.scatter(pxl_by_pxl_output_csv['LON'][idx], pxl_by_pxl_output_csv['LAT'][idx], marker='^', s=30, color='k')
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('$\mathit{s _{%s}}$' %(label_tag), fontsize=label_size)

    c2 = ax2.scatter(OCO_class.lon2d[mask], OCO_class.lat2d[mask], 
                     c=getattr(OCO_class, inter_compare)[:,:,0][mask], s=10,
                     cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('$\mathit{i _{%s}}$' %(label_tag), fontsize=label_size)
    
    lonlat_interval = 0.1 if (lon_2d.max()-lon_2d.min())<1 else 0.2
    for ax, label in zip([ax1, ax2], ['(a)', '(b)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='r')
        ax.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, lonlat_interval)))
        ax.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, lonlat_interval)))
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')

    save_figure(f, f'{file_tag}_{slope_compare}_test.png', img_dir, pad=0.5)


def continuum_fp_compare_plot(o1, o2, o3, 
                              img, wesn, lon_dom, lat_dom, 
                              lon_2d, lat_2d, cth0, 
                              tick_size=12, label_size=14, img_dir='.'):
    f, (ax1, ax2, ax3)=plt.subplots(1, 3, figsize=(24, 7.5))
    for ax in [ax1, ax2, ax3]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='r')
        ax_lon_lat_label(ax, label_size=14, tick_size=12)

    # ax1
    vmin, vmax = 0.03, 0.13
    lev = np.arange(vmin, vmax+1e-7, 0.001)
    rad_to_plot = o1.rad_c3d[:,:,10].copy()
    rad_to_plot[rad_to_plot>vmax] = vmax

    l1b_lon, l1b_lat, l1b_continuum = [], [], []
    for i in range(o1.lon.shape[0]):
        for j in range(o1.lon.shape[1]):
            l1b_lon.append(o1.lon[i, j])
            l1b_lat.append(o1.lat[i, j])
            l1b_continuum.append(o1.l1b[i, j, np.argmin(np.abs(o1.wvl[i, j, :]-o1.lam[10]))])
    cc1 = ax1.contourf(o1.lon2d, o1.lat2d, rad_to_plot, lev, cmap='jet',
                       vmin=vmin, vmax=vmax, extend='both', alpha=1)
    scatter_arg = {'s': 60, 'cmap': 'jet', 'marker': 'o', 'edgecolors': 'k'}
    mask = (o1.lat.flatten()*1e6)>0
    ax1.scatter(np.array(l1b_lon)[mask], np.array(l1b_lat)[mask],
                c=np.array(l1b_continuum)[mask],
                vmin=vmin, vmax=vmax, **scatter_arg)
    cbar1 = f.colorbar(cc1, ax=ax1)
    ax1.set_title(f'{o1.lam[10]:.4f}nm')

    # ax2
    vmin, vmax = 0.0, 0.04
    lev = np.arange(vmin, vmax+1e-7, 0.0005)
    rad_to_plot = o2.rad_c3d[:,:,10].copy()
    rad_to_plot[rad_to_plot>vmax] = vmax

    l1b_lon, l1b_lat, l1b_continuum = [], [], []
    for i in range(o2.lon.shape[0]):
        for j in range(o2.lon.shape[1]):
            l1b_lon.append(o2.lon[i, j])
            l1b_lat.append(o2.lat[i, j])
            l1b_continuum.append(o2.l1b[i, j, np.argmin(np.abs(o2.wvl[i, j, :]-o2.lam[10]))])
    cc2 = ax2.contourf(o2.lon2d, o2.lat2d, rad_to_plot, lev, cmap='jet',
                    vmin=vmin, vmax=vmax, extend='both', alpha=1)
    mask = (o2.lat.flatten()*1e6)>0
    ax2.scatter(np.array(l1b_lon)[mask], np.array(l1b_lat)[mask],
                c=np.array(l1b_continuum)[mask],
                vmin=vmin, vmax=vmax, **scatter_arg)
    cbar2 = f.colorbar(cc2, ax=ax2)
    ax2.set_title(f'{o2.lam[10]:.4f}nm')

    # ax3
    vmin, vmax = 0.00, 0.01
    lev = np.arange(vmin, vmax+1e-7, 0.0001)
    rad_to_plot = o3.rad_c3d[:,:,10].copy()
    rad_to_plot[rad_to_plot>vmax] = vmax

    l1b_lon, l1b_lat, l1b_continuum = [], [], []
    for i in range(o1.lon.shape[0]):
        for j in range(o3.lon.shape[1]):
            l1b_lon.append(o3.lon[i, j])
            l1b_lat.append(o3.lat[i, j])
            l1b_continuum.append(o3.l1b[i, j, np.argmin(np.abs(o3.wvl[i, j, :]-o3.lam[10]))])
    cc3 = ax3.contourf(o3.lon2d, o3.lat2d, rad_to_plot, lev, cmap='jet',
                    vmin=vmin, vmax=vmax, extend='both', alpha=1)
    mask = (o3.lat.flatten()*1e6)>0
    ax3.scatter(np.array(l1b_lon)[mask], np.array(l1b_lat)[mask],
                c=np.array(l1b_continuum)[mask],
                vmin=vmin, vmax=vmax, **scatter_arg)
    cbar3 = f.colorbar(cc3, ax=ax3)
    ax3.set_title(f'{o3.lam[10]:.4f}nm')

    for cbar in [cbar1, cbar2, cbar3]:
        cbar.set_label('Radiance', fontsize=16)
    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')

    save_figure(f, 'continuum_fp_compare.png', img_dir)


def cld_position_func(cfg_name):
    cldfile = f'../simulation/data/{cfg_name}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        lon_cld = f['lon'][...]
        lat_cld = f['lat'][...]
        cth = f[f'mod/cld/cth_ipa'][...]
        cld_list = cth>0
    return lon_cld, lat_cld, cld_list

def cld_dist_calc(cfg_name, o1, slope_compare):
    _, _, cld_list = cld_position_func(cfg_name)
    cld_position = np.argwhere(cld_list)
    cloud_dist = np.zeros_like(getattr(o1, slope_compare)[:,:,0])
    for j in range(cloud_dist.shape[1]):
        for i in range(cloud_dist.shape[0]):
            if cld_list[i, j] == 1:
                cloud_dist[i, j] = 0
            else:
                min_ind = np.argmin(np.sqrt(np.sum((cld_position-np.array([i, j]))**2, axis=1)))
                cld_x, cld_y = cld_position[min_ind]
                cloud_dist[i, j] = geopy.distance.distance((o1.lat2d[cld_x, cld_y], o1.lon2d[cld_x, cld_y]), 
                                                           (o1.lat2d[i, j], o1.lon2d[i, j])).km
    output = np.array([o1.lon2d, o1.lat2d, cloud_dist,])
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], -1).T, columns=['lon', 'lat', 'cld_dis', ])
    cld_slope_inter.to_pickle(f'{cfg_name}_cld_distance.pkl')

def weighted_cld_dist_calc(cfg_name, o1, slope_compare):
    lon_cld, lat_cld, cld_list = cld_position_func(cfg_name)
    cld_position = np.argwhere(cld_list)
    cld_latlon = np.array([[lat_cld[i, j], lon_cld[i, j]] for i, j in cld_position])
    weighted_cloud_dist = np.zeros_like(getattr(o1, slope_compare)[:,:,0])
    for j in range(weighted_cloud_dist.shape[1]):
        for i in range(weighted_cloud_dist.shape[0]):
            if cld_list[i, j] == 1:
                weighted_cloud_dist[i, j] = 0
            else:
                point = np.array([o1.lat2d[i, j], o1.lon2d[i, j]])
                distances = haversine_vector(point, cld_latlon, unit=Unit.KILOMETERS, comb=True)
                # Calculate the inverse distance weights
                weights = 1 / distances**2  
                # Calculate the weighted average distance
                weighted_cloud_dist[i, j] = np.sum(distances * weights) / np.sum(weights)
    output = np.array([o1.lon2d, o1.lat2d, weighted_cloud_dist,])
    print('output shape:', output.shape)
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], -1).T, columns=['lon', 'lat', 'cld_dis', ])
    cld_slope_inter.to_pickle(f'{cfg_name}_weighted_cld_distance.pkl')   

def valid_data(x, y):
    # Remove invalid data points
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    return x[mask], y[mask]

def plot_errorbar(x, y_med, y_std, ax, color='r'):
    ax.errorbar(x, y_med, yerr=y_std, marker='s', markersize=3, color=color,
                linewidth=2, linestyle='', ecolor='skyblue')

def exp_decay_func(x, a, b):
     return a * np.exp(-b * x)

def heatmap_xy_3(x, y, ax, tag=''):
    # Remove invalid values
    x, y = valid_data(x, y)
    # cloud distance
    interval = 1/2
    start = 1
    # Scatter plot for valid data points
    ax.scatter(x[x>=start], y[x>=start], s=1, color='k')
    sns.kdeplot(x=x, y=y, cmap='hot_r', n_levels=20, fill=True, ax=ax, alpha=0.65)
    # Calculate value median of each cloud level
    cld_levels = np.arange(start, 50, interval)
    value_med, value_std = np.zeros(len(cld_levels)-1), np.zeros(len(cld_levels)-1)
    for i in range(len(cld_levels)-1):
        select = np.logical_and(x>=cld_levels[i], x < cld_levels[i+1])
        if select.sum()>5:
            value_med[i] = np.percentile(y[select], 50)
            value_std[i] = np.percentile(y[select], 75)-np.percentile(y[select], 25)
        else:
            value_med[i] = np.nan
            value_std[i] = np.nan
    cld_list = (cld_levels[:-1] + cld_levels[1:])/2
    
    plot_errorbar(cld_list, value_med, value_std, ax, color='r')
    
    val_mask = ~(np.isnan(value_med) | np.isnan(value_std) | np.isinf(value_med) | np.isinf(value_std))
    temp_r2 = 0
    cld_val = cld_list[val_mask]
    cld_min_list = [1+0.5*i for i in range(3)] if cld_val.min()<=2 else [cld_val.min().round(0)+0.25*(i-1) for i in range(3)] 
    cld_max_start = 10 if cld_val.min()<=2 else  15 
    for cld_min in cld_min_list[:1]:
        for cld_max in np.arange(cld_max_start, 50, 1):
            mask = np.logical_and(np.logical_and(cld_val>=cld_min, cld_val<=cld_max), value_std[val_mask]>0)
            xx = cld_val[mask]
            yy = value_med[val_mask][mask]
            if len(yy) > 0:
                popt, pcov = curve_fit(exp_decay_func, xx, yy, bounds=([1e-5, 1e-5], [15, 15,]),
                                       p0=(0.1, 0.7),
                                       maxfev=5000,
                                       sigma=value_std[val_mask][mask], 
                                       absolute_sigma=True,)
                residuals = yy - exp_decay_func(xx, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((yy-np.mean(yy))**2)
                r_squared = 1 - (ss_res / ss_tot)
                if r_squared > temp_r2:
                    temp_r2 = r_squared
                else:
                    break
    perr = np.sqrt(np.diag(pcov))
    e_fold_dist = 1/popt[1]
    e_fold_dist_err = perr[1]/(popt[1]**2)
    amp_str = r'$\mathrm{a_{%s}}$' %(tag)
    d_ef_tag = r'$\mathrm{d_{%s}}$' %(tag)
    plot_xx = np.arange(0, cld_list.max()+0.75, 0.5)
    ax.plot(plot_xx, exp_decay_func(plot_xx, *popt), '--', color='limegreen', 
            label='fit: {} = {:.3f} $\pm$ {:.3f}\n     {} =   {:.2f} $\pm$ {:.2f}'.format(amp_str, popt[0], perr[0], d_ef_tag, e_fold_dist, e_fold_dist_err), linewidth=3.5)
    ax.legend(fontsize=13)
    return popt, perr

def fitting_3bands(cloud_dist, o1, o2, o3, rad_3d_compare, rad_clr_compare, 
                   slope_compare, inter_compare, region_mask,
                   img_dir='.'):

    return_list = []
    fig, ((ax11, ax12), 
          (ax21, ax22),
          (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    label_size = 16
    tick_size = 13

    ax_list = np.array([(ax11, ax12), 
                        (ax21, ax22),
                        (ax31, ax32)])
    for i in range(3):
        oco_band = [o1, o2, o3][i]
        rad_3d = getattr(oco_band, rad_3d_compare)[:,:, -1].flatten()
        rad_clr = getattr(oco_band, rad_clr_compare)[:,:, -1].flatten()
        mask = np.logical_and(np.logical_and(cloud_dist > 0, rad_3d>rad_clr), region_mask)
        slope = getattr(oco_band, slope_compare)[:,:,0].flatten()
        inter = getattr(oco_band, inter_compare)[:,:,0].flatten()
        ax1, ax2 = ax_list[i]
        (slope_a, slope_b), (slope_a_unc, slope_b_unc) = heatmap_xy_3(cloud_dist[mask], slope[mask], ax1, tag='s')
        (inter_a, inter_b), (inter_a_unc, inter_b_unc) = heatmap_xy_3(cloud_dist[mask], inter[mask], ax2, tag='i')
        return_list.append((slope_a, slope_b, inter_a, inter_b))

    ax11.set_ylim(-0.1, 0.3)
    ax12.set_ylim(-0.05, 0.2)
    ax21.set_ylim(-0.1, 0.2)
    ax22.set_ylim(-0.05, 0.2)
    ax31.set_ylim(-0.1, 0.2)
    ax32.set_ylim(-0.05, 0.2)
    cld_low, cld_max = 0, 45

    label_list = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, ax in enumerate(ax_list.flatten()):
        ax.set_xlim(cld_low, cld_max)
        label_text = f'({label_list[i]})'
        ax.set_xlabel('Cloud distance (km)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(0, 0, xmax, linestyle='--', color='white')
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, fontsize=label_size, color='k')
    
    for ax_l, ax_r, band_tag in zip([ax11, ax21, ax31], [ax12, ax22, ax32], ['O_2-A', 'WCO_2', 'SCO_2']):
        ax_l.set_ylabel('$\mathit{s _{%s}}$' %(band_tag), fontsize=label_size)
        ax_r.set_ylabel('$\mathit{i _{%s}}$' %(band_tag), fontsize=label_size)
    save_figure(fig, f'all_band_{slope_compare.split("_")[-1]}.png', img_dir,)# pad=5.0)
    plt.close()
    plt.clf()
    return return_list

def fitting_3bands_with_weighted_dis(cloud_dist, o1, o2, o3, 
                                     rad_3d_compare, rad_clr_compare, slope_compare, inter_compare, region_mask,
                                     img_dir='.'):
    return_list, return_list_err = [], []
    fig, ((ax11, ax12), 
          (ax21, ax22),
          (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    label_size = 16
    tick_size = 13

    ax_list = np.array([(ax11, ax12), 
                        (ax21, ax22),
                        (ax31, ax32)])
    for i in range(3):
        oco_band = [o1, o2, o3][i]
        rad_3d = getattr(oco_band, rad_3d_compare)[:,:, -1].flatten()
        rad_clr = getattr(oco_band, rad_clr_compare)[:,:, -1].flatten()
        mask = np.logical_and(np.logical_and(cloud_dist > 0, rad_3d>rad_clr), region_mask)
        
        slope = getattr(oco_band, slope_compare)[:,:,0].flatten()
        inter = getattr(oco_band, inter_compare)[:,:,0].flatten()

        ax1, ax2 = ax_list[i]
        (slope_a, slope_b), (slope_a_unc, slope_b_unc) = heatmap_xy_3(cloud_dist[mask], slope[mask], ax1, tag='s')
        (inter_a, inter_b), (inter_a_unc, inter_b_unc) = heatmap_xy_3(cloud_dist[mask], inter[mask], ax2, tag='i')
        return_list.append((slope_a, slope_b, inter_a, inter_b))
        return_list_err.append((slope_a_unc, slope_b_unc, inter_a_unc, inter_b_unc))

    ax11.set_ylim(-0.1, 0.2)
    ax12.set_ylim(-0.05, 0.25)
    ax21.set_ylim(-0.05, 0.15)
    ax22.set_ylim(-0.05, 0.25)
    ax31.set_ylim(-0.05, 0.15)
    ax32.set_ylim(-0.05, 0.2)

    cld_low, cld_max = 0, 45
    label_list = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, ax in enumerate(ax_list.flatten()):
        ax.set_xlim(cld_low, cld_max)
        label_text = f'({label_list[i]})'
        ax.set_xlabel('$\mathrm{D_e}$ (km)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(0, 0, xmax, linestyle='--', color='white')
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, fontsize=label_size, color='k')
    
    for ax_l, ax_r, band_tag in zip([ax11, ax21, ax31], [ax12, ax22, ax32], ['O_2-A', 'WCO_2', 'SCO_2']):
        ax_l.set_ylabel('$\mathit{s _{%s}}$' %(band_tag), fontsize=label_size)
        ax_r.set_ylabel('$\mathit{i _{%s}}$' %(band_tag), fontsize=label_size)
    save_figure(fig, f'all_band_weighted_dis_{slope_compare.split("_")[-1]}.png', img_dir,)# pad=5.0)
    plt.close()
    plt.clf()
    return return_list, return_list_err


            
def ax_index_label(ax, label, label_size):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin+0.0*(xmax-xmin), ymin+1.035*(ymax-ymin), label, fontsize=label_size, color='k')

if __name__ == "__main__":
    now = time.time()
    main()
    print(f'{(time.time()-now)/60:.3f} min')