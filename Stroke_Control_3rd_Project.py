#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:50:37 2023

@author: sara
"""

import pickle
import mne
import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import meta_functions_HC as function_hc
from mne.channels import make_standard_montage
from scipy.signal import find_peaks
import statistics

from scipy.signal import medfilt
#%%
evokeds_pahse_effecft_all = []
epochs_data_dir = "/home/sara/data/Third part/epochs/"
#epochs_data_dir ="/home/sara/data/Second part/1st_v/epochs"
epochs_files = Path(epochs_data_dir).glob('*epo.fif*')
#epochs_files = Path(epochs_data_dir).glob('*BuUl_pre3_old2R_epo*')
plt.close('all')

Save_folder = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/"

dictionary_bad_channels = { # AmWo
                           'AmWo_post1L': ['FT9', 'TP9', 'Iz'],
                           'AmWo_post3_old1L': ['Iz', 'FT9', 'T7', 'F6', 'P5', 'C1'],
                           'AmWo_post5_old1L': ['TP10', 'FT10', 'Iz', 'PO7', 'TP9', 'FT9', 'Fpz'],
                           'AmWo_pre1L': ['T7', 'Iz', 'TP8', 'P8', 'TP9', 'FT9', 'Fp1'],
                           'AmWo_pre4_old1L': ['TP9', 'Iz', 'FT9', 'T7', 'AF8', 'P6'],
                           
                         
                            # BuUl
                           'BuUl_post2R': ['Iz', 'Fz', 'P7'],
                           'BuUl_post4R': ['Iz', 'FT9', 'TP9', 'T8'],
                           'BuUl_post5_old1R':['TP10', 'T8', 'TP9', 'FT9'],
                           'BuUl_pre2R': ['P7', 'Fpz', 'P5', 'AF8', 'C3', 'TP7', 'TP9', 'FT7', 'PO8', 'O2', 'T7', 'Iz'],
                           'BuUl_pre3_old2R': ['T7', 'FC2', 'O1', 'Iz', 'PO3', 'PO7', 'P7', 'CP4', 'C1', 'Fpz', 'Fp1','FT10', 'F3'],
                           
                            # EiHe           
                           'EiHe_post1_2R': ['Fz', 'Iz'],
                           'EiHe_post4_old1R': ['Fpz', 'Fp2', 'T8', 'F1', 'Fp1', 'Fz', 'T7', 'AF8', 'PO3', 'FC3', 'C3', 'PO7', 'F7'],
                           'EiHe_post5_old1R': ['T8', 'FT8', 'T7', 'FT7', 'Iz'],
                           'EiHe_pre2R': ['Iz', 'FT9', 'T7', 'F7', 'FT7', 'AF7', 'T8', 'P8'],
                           'EiHe_pre3_old1R': ['Fp1', 'Fp2', 'Fpz', 'F3', 'AF8', 'T7', 'Iz', 'F7', 'O1', 'PO7', 'C3', 'FT7', 'CP5'],
                           
                            # FuMa
                           'FuMa_post2L': ['TP9', 'TP7', 'Iz', 'PO8','FT10', 'TP10', 'Iz'],
                           'FuMa_post4L': ['AF7', 'FT9'],
                           'FuMa_post6_old1L': ['TP7', 'TP9', 'FC6', 'FT10', 'T8', 'FT8', 'C6', 'FC6', 'Iz', 'P7', 'P5'],
                           'FuMa_pre1L': ['Iz', 'TP7', 'TP9', 'Fp1', 'Fp2', 'Fpz', 'AF8', 'Oz', 'TP10', 'FC5', 'AF4', 'F6', 'AF7', 'FT7', 'F7', 'CP3', 'PO4', 'C2', 'CP2'],
                           'FuMa_pre4_old1L': ['F3', 'P6', 'C1', 'AF8', 'P8', 'FT10', 'FT9', 'TP9'],
                           

                            # GrMa
                           'GrMa_pre1L': ['C3', 'CP3', 'FC3', 'FT10', 'P8', 'Fp1', 'F7', 'FT7', 'C4', 'FT9', 'TP10', 'PO7', 'Iz'],
                           'GrMa_post2L': ['FT9', 'T7', 'Iz', 'AF8', 'F6', 'FT9', 'TP9'],
                           'GrMa_post4_old1L': ['Oz', 'F3', 'P6', 'Fp2', 'Fp1', 'Fpz', 'AF8'],
                           'GrMa_post6_old1L': ['FT9', 'TP9', 'T7', 'PO7', 'F8'],
                           'GrMa_pre3L': ['FC5', 'Fp1'],
                           
                            # GuWi                        
                           'GuWi_post1R': ['TP8', 'TP9'],
                           'GuWi_post3R': ['TP10', 'Iz', 'P8'],
                           'GuWi_post6_old1R': ['TP8', 'CP6'],
                           'GuWi_pre1_old1R': ['T8', 'FC5', 'FT9'],
                           'GuWi_pre4R': ['FT9', 'FT10'],
                           
                              
                            # KaBe
                           'KaBe_post1L': ['TP8', 'T8', 'Iz', 'TP10', 'FT10', 'FT8', 'T7'],
                           'KaBe_post4_old1L': ['Iz', 'AF8'], 
                           'KaBe_post6_old1L': ['TP9', 'FT9', 'FT7', 'T7', 'TP10', 'FT10', 'TP8', 'T8', 'P7', 'PO7', 'O1', 'P5', 'PO3'],
                           'KaBe_pre2L': ['TP9', 'FT9', 'FT10', 'Iz', 'Oz', 'PO8', 'T7', 'PO7', 'O1', 'F4'], 
                           'KaBe_pre4L': ['Fp1', 'Fp2', 'Fpz', 'Fz', 'AF4', 'P4', 'F1', 'TP9', 'TP7', 'AF8', 'O2',  'O1', 'PO3', 'Iz', 'F3', 'FC5', 'Oz', 'PO8'],

                            # MeRu 
                           'MeRu_post2R':['Fp1', 'Fpz', 'FT9', 'TP9', 'FC5', 'AF7'],
                           'MeRu_post4_old1R': ['PO7','TP7', 'Iz'],
                           'MeRu_post5_old1R': ['PO7', 'TP10', 'Fpz'],
                           'MeRu_pre1R': ['Iz'],
                           'MeRu_pre4R': ['PO7'],

                            # SoFa
                           'SoFa_pre1_old1L': ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF8', 'Cz', 'T8'],
                           'SoFa_post1L': ['FT9', 'TP9', 'Iz', 'P8', 'T7', 'F3', 'AF8'],
                           'SoFa_post3_old1L': ['PO7'],
                           'SoFa_post5L': ['PO7', 'PO4', 'Oz', 'Iz', 'TP9', 'TP7'],
                           'SoFa_pre4L': ['Iz', 'FT9', 'TP9','F3', 'F7', 'AF8', 'F6','T7'],
                           
                            # WiLu
                           'WiLu_pre2L': ['TP9', 'TP10', 'Iz', 'O2', 'C3', 'AF8', 'P5', 'FT9', 'AF4', 'Fp2', 'F8', 'FT10', 'PO8', 'O2', 'P8', 'T8', 'TP8', 'FT8', 'AF3', 'FC4'],
                           'WiLu_pre4_old1L': ['FT9', 'Fz', 'T7', 'Iz', 'TP10', 'Fpz', 'P5', 'FC3', 'F4', 'F7', 'PO8', 'Fp1', 'P8', 'FT10', 'Oz', 'O2', 'FT10', 'TP10', 'TP8', 'F8', 'FT8', 'T8', 'PO7', 'O1', 'Oz', 'Pz', 'P6'],
                           'WiLu_post1L': ['AF8', 'T7', 'C1', 'Fp1', 'FT9', 'Iz', 'F3', 'P8'], 
                           'WiLu_post3_old1L': ['Fpz', 'AF8', 'Fp1', 'Fp2', 'AF7', 'AF4', 'O1'], 
                           'WiLu_post5_old1L': ['FT10', 'TP10', 'O1', 'PO7', 'TP9', 'TP8', 'P8', 'P6', 'PO8', 'T8', 'Oz', 'Iz', 'O2', 'PO3', 'PO4', 'POz', 'C6', 'FT9', 'T7', 'TP7', 'P7', 'P5'],
                                               
}

"""14 subjects are selected in the end, when 60 % of the data is available (3/5)"""

stim_artifact_subs = {'AmWo_pre1L'  : [-40, 20], 
                      'EiHe_post4_old1R' : [-30, 10],
                      'GrMa_post2L' : [-40, 20],
                      'SoFa_post1L': [-40, 20],
                      'WiLu_pre2L': [-50, 10],
                      'WiLu_pre4_old1L': [-20, 5], 
                      'AmWo_post3_old1L': [-30, 20], 
                      'GrMa_pre1L': [-30, 10], 
                      'KaBe_pre2L': [-20, 20],
                      'KaBe_pre4L': [-40, 20]
                      }

        # Amwo [-30, 20]
        # KaBe_pre2L [-20, 10]
        # SoFa_post1L[-40,20]
        # EiHe_post4 [-30, 10]
        # GrMa_post2L [-40, 20]

evokeds_all_L = {}
evokeds_all_R = {}
sub_names_L = {}
sub_names_R = {}
for _,time_point in enumerate(['v2', 'v3', 'v4', 'v5', 'v6']):
    evokeds_all_R[str(time_point)] = []
    evokeds_all_L[str(time_point)] = []
    sub_names_L[str(time_point)] = []
    sub_names_R[str(time_point)] = []


for f in epochs_files:
    plt.close('all')
    subject_ID = f.parts[-1][0:-8]
    print(f.parts[-1])
    

    if subject_ID in dictionary_bad_channels:
        epochs = mne.read_epochs(f, preload= True)
        epochs = epochs.set_eeg_reference(ref_channels='average')      
        evokeds = epochs.average()     
        montage = make_standard_montage('standard_1005')
        epochs = epochs.set_montage(montage)
        
        
        
        all_times = np.arange(0, 0.5, 0.01)
        topo_plots = evokeds.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11))
        ERP_plots = evokeds.plot(spatial_colors = True, gfp = True) 
        ERP_plots.set_size_inches((20, 8))
       
        epochs.info['bads'] = dictionary_bad_channels[subject_ID]
        epochs_clean = epochs.interpolate_bads(reset_bads=True, mode='accurate')

        
        # cubic interpolation for some subjects that have sth like a TMS artifact
        for sub_name, sub_name_v in enumerate(stim_artifact_subs): 
            if (subject_ID == sub_name_v):
                print(sub_name_v)
                epochs_clean = function_hc.cubic_interp(epochs_clean, win = stim_artifact_subs[sub_name_v])
                epochs_clean = epochs_clean
    
        evokeds_clean = epochs_clean.average()    
        all_times = np.arange(0, 0.5, 0.01)
        topo_plots = evokeds_clean.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11))
        ERP_plots = evokeds_clean.plot(spatial_colors = True, gfp = True) 
        ERP_plots.set_size_inches((20, 8))
        topo_plots_senors = evokeds_clean.plot_topomap(np.arange(0, 0.4, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
        topo_plots_senors.savefig(Save_folder+ 'figs/each_sub/' + f'{subject_ID}' + '_topo'    + '.svg', overwrite = True) 
        ERP_plots.savefig(Save_folder+ 'figs/each_sub/' + f'{subject_ID}' + '_erp'    + '.svg', overwrite = True) 
    
        

    
        if subject_ID[-1] == 'R':
            if subject_ID[5:9] == 'pre1' or  subject_ID[5:9] == 'pre2':
                evokeds_all_R[str('v2')].append(evokeds_clean)
                sub_names_R[str('v2')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v2/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:9] == 'pre3' or  subject_ID[5:9] == 'pre4':
                evokeds_all_R[str('v3')].append(evokeds_clean)
                sub_names_R[str('v3')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v3/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:10] == 'post1' or  subject_ID[5:10] == 'post2':
                evokeds_all_R[str('v4')].append(evokeds_clean)
                sub_names_R[str('v4')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v4/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:10] == 'post3' or  subject_ID[5:10] == 'post4':
                evokeds_all_R[str('v5')].append(evokeds_clean)
                sub_names_R[str('v5')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v5/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:10] == 'post5' or  subject_ID[5:10] == 'post6':
                evokeds_all_R[str('v6')].append(evokeds_clean)
                sub_names_R[str('v6')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v6/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
  
        elif subject_ID[-1] == 'L':
            if subject_ID[5:9] == 'pre1' or  subject_ID[5:9] == 'pre2':
                evokeds_all_L[str('v2')].append(evokeds_clean)
                sub_names_L[str('v2')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v2/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:9] == 'pre3' or  subject_ID[5:9] == 'pre4':
                evokeds_all_L[str('v3')].append(evokeds_clean)
                sub_names_L[str('v3')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v3/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')

                
            elif subject_ID[5:10] == 'post1' or  subject_ID[5:10] == 'post2':
                evokeds_all_L[str('v4')].append(evokeds_clean)
                sub_names_L[str('v4')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v4/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')

                    
            elif subject_ID[5:10] == 'post3' or  subject_ID[5:10] == 'post4':
                evokeds_all_L[str('v5')].append(evokeds_clean)
                sub_names_L[str('v5')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v5/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')

                
            elif subject_ID[5:10] == 'post5' or  subject_ID[5:10] == 'post6':
                evokeds_all_L[str('v6')].append(evokeds_clean)
                sub_names_L[str('v6')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v6/' + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
              
                
             
Evoked_GrandAv_R = {} 
Evoked_GrandAv_L = {}           
for _,time_point in enumerate(['v2', 'v3', 'v4', 'v5', 'v6']):
    Evoked_GrandAv_R[str(time_point)] =  mne.grand_average(evokeds_all_R[str(time_point)]) 
    Evoked_GrandAv_L[str(time_point)] = mne.grand_average(evokeds_all_L[str(time_point)])             
             
             

    # Right side plots
    ERP_plots_R = Evoked_GrandAv_R[str(time_point)].plot(spatial_colors = True, gfp = True) 
    ERP_plots_R.set_size_inches((20, 8))
    Evoked_GrandAv_R_fig = Evoked_GrandAv_R[str(time_point)].crop(-0.015, 0.5).plot_joint(times= [0.040, 0.060, 0.120, 0.150], ts_args = dict( ylim=dict(eeg=[-2, 2]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-2, 2], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
    topo_plots_senors_r = Evoked_GrandAv_R[str(time_point)].plot_topomap(np.arange(0, 0.4, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
    


    # Left Side plots
    ERP_plots_L = Evoked_GrandAv_L[str(time_point)].plot(spatial_colors = True, gfp = True) 
    ERP_plots_L.set_size_inches((20, 8))
    topo_plots_senors_L = Evoked_GrandAv_L[str(time_point)].plot_topomap(np.arange(0, 0.4, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
    Evoked_GrandAv_L_fig = Evoked_GrandAv_L[str(time_point)].crop(-0.015, 0.5).plot_joint(times=[0.040, 0.060, 0.120, 0.150], ts_args = dict( ylim=dict(eeg=[-2, 2]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-2, 2], units = dict(eeg='µV')  , sphere=(0.00, 0.00, 0.00, 0.11)))
    

    # Save Figures
    topo_plots_senors_r.savefig(Save_folder + 'figs/' +'R_' + str(time_point) + '_topo_plots_senors_patients_'  + '.svg', overwrite = True) 
    Evoked_GrandAv_R_fig.savefig(Save_folder+ 'figs/' +'R_' + str(time_point) + '_Evoked_GrandAv_patients_'     + '.svg', overwrite = True) 
    topo_plots_senors_L.savefig(Save_folder + 'figs/' +'L_' + str(time_point) + '_topo_plots_senors_patients_'  + '.svg', overwrite = True) 
    Evoked_GrandAv_L_fig.savefig(Save_folder+ 'figs/' +'L_' + str(time_point) + '_Evoked_GrandAv_patients_'     + '.svg', overwrite = True) 








save_folder_peak = '/home/sara/data/Third part/epochs_manually_rejected/V2/peak_amp_compare/'
# saving evoked files
mne.evoked.write_evokeds(save_folder_peak + 'ST_V2_L_ave.fif', Evoked_GrandAv_L[str('v2')], overwrite = True)
mne.evoked.write_evokeds(save_folder_peak + 'ST_V2_R_ave.fif', Evoked_GrandAv_R[str('v2')], overwrite = True)

# Left
p1_l_v2 = Evoked_GrandAv_L[str('v2')].plot_topomap(times=[0.043], average=0.016,  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
n1_l_v2 = Evoked_GrandAv_L[str('v2')].plot_topomap(times=[0.06],  average=0.021,  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
p2_l_v2 = Evoked_GrandAv_L[str('v2')].plot_topomap(times=[0.195], average=0.031,  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
# Saving
p1_l_v2.savefig(save_folder_peak  + 'p1_l_v2.svg', overwrite = True)
n1_l_v2.savefig(save_folder_peak  + 'n1_l_v2.svg', overwrite = True)
p2_l_v2.savefig(save_folder_peak  + 'p2_l_v2.svg', overwrite = True)
# Right
p1_r_v2 = Evoked_GrandAv_R[str('v2')].plot_topomap(times=[0.043], average=0.016,  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
n1_r_v2 = Evoked_GrandAv_R[str('v2')].plot_topomap(times=[0.06],  average=0.021,  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
p2_r_v2 = Evoked_GrandAv_R[str('v2')].plot_topomap(times=[0.195], average=0.031,  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
# Saving
p1_r_v2.savefig(save_folder_peak  + 'p1_r_v2.svg', overwrite = True)
n1_r_v2.savefig(save_folder_peak  + 'n1_r_v2.svg', overwrite = True)
p2_r_v2.savefig(save_folder_peak  + 'p2_r_v2.svg', overwrite = True)


win_erp0 = [35, 50]   #P1
win_erp1 = [50, 70]   #N1
win_erp2 = [180, 210] #p2
win_erps = np.array([win_erp0, win_erp1, win_erp2])

function_hc.plot_three_evoked_potentials(Evoked_GrandAv_L[str('v2')], win_erps, 'ST_L.avg', save_folder_peak)
function_hc.plot_three_evoked_potentials(Evoked_GrandAv_R[str('v2')], win_erps, 'ST_R', save_folder_peak)   



with open(str(save_folder_peak) + 'ST_ERP_R_V2_V6.p', 'wb') as fp:
    pickle.dump(Evoked_GrandAv_R, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(str(save_folder_peak) + 'ST_ERP_L_V2_V6.p', 'wb') as fp:
    pickle.dump(Evoked_GrandAv_L, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    



grand_average_v2 = mne.grand_average([Evoked_GrandAv_L[str('v2')], Evoked_GrandAv_R[str('v2')]])
grand_average_v2.crop(tmin = -.015, tmax=0.35)

grand_average_v2_plot = grand_average_v2.plot_joint(times = [0.065],  ts_args = dict( ylim=dict(eeg=[-1.5, 1.5]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_v2_plot.set_size_inches((7, 5))

grand_average_v2_plot.savefig(save_folder_peak  + 'grand_average_v2_plot.svg', overwrite = True)






#############
#%%
Evoked_GrandAv_L_v2 = mne.grand_average(evokeds_all_L[str('v2')])
Evoked_GrandAv_L_v2.crop(tmin = -0.06, tmax=0.5)
Evoked_GrandAv_L_v2.info['bads'] = ['FT10', 'FT8', 'F8', 'TP8', 'P8', 'T8', 'PO8', 'FT10', 'Fpz', 'TP9', 'O1', 'PO7', 'TP7', 'P7', 'T7']
Evoked_GrandAv_L_v2 = Evoked_GrandAv_L_v2.interpolate_bads(reset_bads=True, mode='accurate')
grand_average_v2_plot_l = Evoked_GrandAv_L_v2.plot_joint(times = [0.04, 0.07, 0.12, 0.18],  ts_args = dict( ylim=dict(eeg=[-2.5, 2.5]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_v2_plot_l.set_size_inches((7, 6))
grand_average_v2_plot_l.savefig(save_folder_peak  + 'grand_average_v2_plot_l.png', overwrite = True)


Evoked_GrandAv_R_v2 = mne.grand_average(evokeds_all_R[str('v2')])
Evoked_GrandAv_R_v2.crop(tmin = -0.06, tmax=0.5)
Evoked_GrandAv_R_v2.info['bads'] = ['TP9', 'PO7', 'O1', 'Oz']
Evoked_GrandAv_R_v2 = Evoked_GrandAv_R_v2.interpolate_bads(reset_bads=True, mode='accurate')
grand_average_v2_plot_r = Evoked_GrandAv_R_v2.plot_joint(times = [0.04, 0.06, 0.12, 0.18],  ts_args = dict( ylim=dict(eeg=[-2.5, 2.5]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_v2_plot_r.set_size_inches((7, 6))
grand_average_v2_plot_r.savefig(save_folder_peak  + 'grand_average_v2_plot_r.png', overwrite = True)

#%%

import meta_functions_HC as function_hc

with open(str(save_folder_peak) + 'evokeds_all_L_hc.p', 'rb') as fp:
    evokeds_all_L_hc = pickle.load(fp)

with open(str(save_folder_peak) + 'evokeds_all_R_hc.p', 'rb') as fp:
    evokeds_all_R_hc = pickle.load(fp)
    



Evoked_GrandAv_L_hc = mne.grand_average(evokeds_all_L_hc)
Evoked_GrandAv_L_hc.crop(tmin = -0.06, tmax=0.5)
Evoked_GrandAv_L_hc.info['bads'] = ['FT10', 'FT8', 'F8', 'TP8', 'P8', 'T8', 'PO8', 'FT10', 'Fpz', 'TP9', 'O1', 'PO7', 'TP7', 'P7', 'T7']
Evoked_GrandAv_L_hc = Evoked_GrandAv_L_hc.interpolate_bads(reset_bads=True, mode='accurate')
grand_average_hc_plot_l = Evoked_GrandAv_L_hc.plot_joint(times = [0.04, 0.07, 0.12, 0.18],  ts_args = dict( ylim=dict(eeg=[-2.5, 2.5]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_hc_plot_l.set_size_inches((7, 6))
grand_average_hc_plot_l.savefig(save_folder_peak  + 'Evoked_GrandAv_L_hc_plot.png', overwrite = True)



Evoked_GrandAv_R_hc = mne.grand_average(evokeds_all_R_hc)
Evoked_GrandAv_R_hc.crop(tmin = -0.06, tmax=0.5)
Evoked_GrandAv_R_hc.info['bads'] = ['TP9', 'PO7', 'O1', 'Oz']
Evoked_GrandAv_R_hc = Evoked_GrandAv_R_hc.interpolate_bads(reset_bads=True, mode='accurate')
grand_average_hc_plot_r = Evoked_GrandAv_R_hc.plot_joint(times = [0.04, 0.06, 0.12, 0.18],  ts_args = dict( ylim=dict(eeg=[-2.5, 2.5]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_hc_plot_r.set_size_inches((7, 6))
grand_average_hc_plot_r.savefig(save_folder_peak  + 'Evoked_GrandAv_R_hc_plot.png', overwrite = True)

#%%    

contra_right = ['C1', 'C3', 'CP1', 'CP3', 'C5', 'CP5']
contra_left =  ['C2', 'C4', 'CP2', 'CP4', 'C6', 'CP6']


function_hc.laterality_Index_errorbar(contra_right, contra_left, evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, save_folder_peak)





#%%

# real clusters
import meta_functions_HC as function_hc

save_folder =  '/home/sara/data/Third part/epochs_manually_rejected/V2_V6/'
exdir_epoch_r = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/Right/"
exdir_epoch_l = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/Left/"


plt.close('all')
win_erp0 = [35, 50]   
win_erp1 = [65, 100]   
win_erp2 = [100, 135] 
win_erp3 = [140, 220] 



labels = ['P1', 'N1', 'N2', 'P2']
win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])
ch_names_r = {}; ch_names_l = {}; pvals_all_r = {}; pvals_all_l = {}; peaks_r  = {}; peaks_l  = {}
mean_peaks_r  = {}; mean_peaks_l  = {}; mask_r = {}; mask_l = {}; t_l = {}; t_r = {}; num_sub_l = 6; num_sub_r = 4


time_points = ['v2', 'v3', 'v4', 'v5', 'v6'] 


#time_points = ['v2'] 


for _,time_point in enumerate(time_points):
    ch_names_r[str(time_point)], pvals_all_r[str(time_point)], t_r[str(time_point)], mask_r[str(time_point)], pos, peaks_r[str(time_point)]  = function_hc.clustering_channels(4, win_erps, exdir_epoch_r + f'{time_point}/',  [0.5, 2.5, 2, 2], labels, 'R_'+ f'{time_point}', save_folder + 'Right/')    
    ch_names_l[str(time_point)], pvals_all_l[str(time_point)], t_l[str(time_point)], mask_l[str(time_point)], pos, peaks_l[str(time_point)]  = function_hc.clustering_channels(6, win_erps, exdir_epoch_l + f'{time_point}/',  [1.5, 0.5, 2, 0.8], labels, 'L_'+ f'{time_point}', save_folder + 'Left/')    

    
  
    
  
#%%

# real clusters
import meta_functions_HC as function_hc

save_folder =  '/home/sara/data/Second part/epochs_manually_rejected/Group_models/'
exdir_epoch_r = "/home/sara/data/Second part/epochs_manually_rejected/Right/"
exdir_epoch_l = "/home/sara/data/Second part/epochs_manually_rejected/Left/"

save_folder_peak = '/home/sara/data/Third part/epochs_manually_rejected/V2/peak_amp_compare/'


win_erp0 = [12, 35]   
win_erp1 = [60, 90]   
win_erp2 = [100, 135] 
win_erp3 = [140, 180] 
win_erps  = np.array([win_erp0, win_erp1, win_erp2, win_erp3])


ch_names_r_hc, pvals_all_r_hc, t_r_hc, mask_r_hc, pos, peaks  = function_hc.clustering_channels(6, win_erps, exdir_epoch_r,  [1.4, 2, 2, 0.8], labels, 'R', save_folder + '/right/')    

ch_names_l_hc, pvals_all_l_hc, t_l_hc, mask_l_hc, pos, peaks  = function_hc.clustering_channels(11, win_erps, exdir_epoch_l, [2, 2.5, 0.8, 0.5], labels, 'L',save_folder + '/left/')    




#%%



## Common channels
maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=5)
maskparam_com = dict(marker='X', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=5)
ch_com_r = {}
for iplot in np.arange(0, 4, 1):
    ch_com_r[str(iplot)] = np.intersect1d(ch_names_r_hc[iplot], ch_names_r[str('v2')][iplot])
ch_com_l = {}
for iplot in np.arange(0, 4, 1):
    ch_com_l[str(iplot)] = np.intersect1d(ch_names_l_hc[iplot], ch_names_l[str('v2')][iplot])


mask_com_l = np.zeros([4, 64])
mask_com_r = np.zeros([4, 64])


time_point = 'v2'
#% Left side
fig, sps = plt.subplots(nrows=3, ncols=4, figsize=(20,8))
for iplot in np.arange(0, 4, 1):
    im = function_hc.topoplot_2d(function_hc.channel_names(), t_l_hc[:, iplot], pos, clim=[-5,5], axes=sps[0,iplot], mask=mask_l_hc[:, iplot], maskparam=maskparam) 
    sps[0,iplot].set_title(f'{labels[iplot]}', fontweight = 'bold', fontsize = 20)
    fig.text(0.2 + 0.2*iplot, 0.67,  f't = {np.round(sum(t_l_hc[np.where(mask_l_hc[:, iplot] == 1)[0]  , iplot]), 2)}',fontsize=14, fontweight = 'bold')
    function_hc.topoplot_2d(function_hc.channel_names(), t_l[str('v2')][:, iplot], pos, clim=[-5,5], axes=sps[1,iplot], mask=mask_l[str('v2')][:, iplot], maskparam=maskparam) 
    fig.text(0.2 + 0.2*iplot, 0.37,  f't = {np.round(sum(t_l[str(time_point)][np.where(mask_l[str(time_point)][:, iplot] == 1)[0]  , iplot]), 2)}',fontsize=14, fontweight = 'bold')
    mask_com_l[iplot, :][function_hc.channel_indices(ch_com_l[str(iplot)])] =1
    function_hc.topoplot_2d(function_hc.channel_names(), t_l[str('v2')][:, iplot], pos, clim=[-5,5], axes=sps[2,iplot], mask=mask_com_l[iplot, :], maskparam=maskparam_com) 
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad = 0.02)
    cb.ax.tick_params(labelsize=12)
    cb.set_label('t-value', rotation = 90)
    plt.show()
fig.savefig(save_folder_peak  + 'hc_vs_c_cluster_l.svg', overwrite = True)    

#% Right side
fig, sps = plt.subplots(nrows=3, ncols=4, figsize=(20,8))
for iplot in np.arange(0, 4, 1):
    function_hc.topoplot_2d(function_hc.channel_names(), t_r_hc[:, iplot], pos, clim=[-5,5], axes=sps[0,iplot], mask=mask_r_hc[:, iplot], maskparam=maskparam) 
    sps[0,iplot].set_title(f'{labels[iplot]}', fontweight = 'bold', fontsize = 20)
    fig.text(0.2 + 0.2*iplot, 0.67, f't = {np.round(sum(t_r_hc[np.where(mask_r_hc[:, iplot] == 1)[0]  , iplot]), 2)}', fontsize=14, fontweight = 'bold')
    function_hc.topoplot_2d(function_hc.channel_names(), t_r[str('v2')][:, iplot], pos, clim=[-5,5], axes=sps[1,iplot], mask=mask_r[str('v2')][:, iplot], maskparam=maskparam) 
    fig.text(0.2 + 0.2*iplot, 0.37,  f't = {np.round(sum(t_r[str(time_point)][np.where(mask_r[str(time_point)][:, iplot] == 1)[0]  , iplot]), 2)}', fontsize=14, fontweight = 'bold')
    mask_com_r[iplot, :][function_hc.channel_indices(ch_com_r[str(iplot)])] =1
    function_hc.topoplot_2d(function_hc.channel_names(), t_r[str('v2')][:, iplot], pos, clim=[-5,5], axes=sps[2,iplot], mask=mask_com_r[iplot, :], maskparam=maskparam_com) 
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad = 0.02)
    cb.ax.tick_params(labelsize=12)
    cb.set_label('t-value', rotation = 90)
fig.savefig(save_folder_peak  + 'hc_vs_c_cluster_r.svg', overwrite = True)    

#%%
  
    
function_hc.N3_cluster_bilaterality(t_r, t_l, pvals_all_r, pvals_all_l, mask_r, mask_l, pos, t_r_hc, t_l_hc, pvals_all_r_hc, pvals_all_l_hc, mask_r_hc, mask_l_hc, save_folder_peak)


  


#%% finding time window for components according to literature



#% P1 N2 contra_ipsi_william
import meta_functions_HC as function_hc

end_time = 1270
contra_right = ['C1', 'C3', 'CP1', 'CP3', 'C5', 'CP5']
contra_left =  ['C2', 'C4', 'CP2', 'CP4', 'C6', 'CP6']
contra_right_ind = function_hc.channel_indices(contra_right)
contra_left_ind = function_hc.channel_indices(contra_left)
function_hc.contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component = 'N2', text = 'P1, N2')



#% N1 contra_ipsi_william
contra_right = ['F1', 'F3', 'FC1', 'FC3']
contra_left =  ['F2', 'F4', 'FC2', 'FC4']
contra_right_ind = function_hc.channel_indices(contra_right)
contra_left_ind = function_hc.channel_indices(contra_left)
function_hc.contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component = 'N1', text = 'N1')



#% P2 contra_ipsi_william
contra_right = ['Cz']
contra_left =  ['Cz']
contra_right_ind = function_hc.channel_indices(contra_right)
contra_left_ind = function_hc.channel_indices(contra_left)
function_hc.contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component = 'P2', text = 'P2')



    
#%%
import meta_functions_HC as function_hc


contra_right = {
    'P1':['C3', 'C5', 'CP3', 'CP5', 'C1', 'CP1'],
    'N1':['F1', 'F3', 'FC1', 'FC3'],
    'N2':ch_names_r_hc[2],
    'P2':['Cz'],
    'P3':['C4', 'C6', 'CP4', 'CP6', 'C2','CP2'] #ipsi
    }

contra_left =  {
    'P1':['C4', 'C6', 'CP4', 'CP6', 'C2','CP2'],
    'N1':['F2', 'F4', 'FC2', 'FC4'],
    'N2':ch_names_l_hc[2],
    'P2':['Cz'],
    'P3':['C3', 'C5', 'CP3', 'CP5', 'C1', 'CP1'] #ipsi
    }






# I'm using P2 vs P3 for contra and ipsi ---> laterality index
time_windows = {'P1':[15, 50], 'N1':[46, 80], 'N2':[100, 135 ],'P2':[150, 220],'P3':[100, 135]}
amp_l_arr_st, amp_r_arr_st, lat_l_arr_st, lat_r_arr_st =  function_hc.amp_latency_6_components_st(evokeds_all_L, evokeds_all_R, contra_right, contra_left, time_windows, save_folder_peak)

lat_l_arr_st_del = lat_l_arr_st[:, 1:, :]
lat_r_arr_st_del = lat_r_arr_st[:, np.array([1, 3]), :]
amp_t_hc, lat_t_hc =  function_hc.amp_latency_6_components_hc(evokeds_all_L_hc, evokeds_all_R_hc, contra_right, contra_left, time_windows, save_folder_peak)
# amp_r ------>  (6 x 4 x 5) =  (component x sub x v2-v6) 


#%% Bar plot statistical analysis

import meta_functions_HC as function_hc
#function_hc.box_plot_n2_HC_st(lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak)
#function_hc.errorbar_plot_n2_HC_st(lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak)




component_number = 0 # P1
p_ind_p1 = function_hc.box_plot_p1_all_time_points_HC_st(component_number, lat_t_hc, lat_l_arr_st_del, lat_r_arr_st_del, save_folder_peak)


component_number = 1 # N1
t_ind , p_ind, t_paired, p_paired = function_hc.bar_plot_n2_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak)

# Our argument is N2 is not formed
component_number = 3 # P2
t_ind , p_ind, t_paired, p_paired = function_hc.bar_plot_p2_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak)



#%%

        
#% Clinical Assessment
clinical_xl =  pd.read_excel('IN-TENS_Clinical_assessments_scores_all .xlsx', sheet_name=None)

subjects_part_ids = ['AmWo', 'FuMa', 'GrMa', 'KaBe', 'SoFa', 'WiLu', 'BuUl', 'EiHe', 'GuWi', 'MeRu']

fuma_score = np.zeros([10, 5]) 

for i, i_n in enumerate(subjects_part_ids):
    
    k = list(clinical_xl[str('Sheet1')][str('part_id')]).index(f'{i_n}')
    
    fuma_score[i, :] = clinical_xl[str('Sheet1')]['fmue'].iloc[np.arange(k, 95,19)]
    
    
fuma_score_swap = fuma_score[:, [0, 2, 1, 3 , 4]]
fuma_score_swap[5, 2] = 21 # just a given value    
time_points = ['v2', 'v3', 'v4', 'v5', 'v6']












p_t = {}
r_t = {}
analysis = ['amp', 'lat']

for i_analysis, analysis in enumerate(analysis):
    p_t[str(analysis)] = np.zeros([len(time_windows), len(time_points)])
    r_t[str(analysis)] = np.zeros([len(time_windows), len(time_points)])
    for i_components, n_components in enumerate(time_windows):         
        for i_time, n_time in enumerate(time_points):   
            if analysis == 'amp':
                r_t[str(analysis)][i_components, i_time], p_t[str(analysis)][i_components, i_time] = scipy.stats.pearsonr(fuma_score_swap[:, i_time], np.concatenate((amp_l_arr_st[i_components,:, i_time ], amp_r_arr_st[i_components,:, i_time])))
            elif analysis == 'lat':   
                r_t[str(analysis)][i_components, i_time], p_t[str(analysis)][i_components, i_time] = scipy.stats.pearsonr(fuma_score_swap[:, i_time], np.concatenate((lat_l_arr_st[i_components,:, i_time ], lat_r_arr_st[i_components,:, i_time])))
                
                
                
                


function_hc.FM_UE_plotting(amp_l_arr_st, amp_r_arr_st, fuma_score_swap, save_folder_peak)



#function_hc.FM_UE_L_R_all(fuma_score_swap, save_folder_peak)

#%%








        
#% Clinical Assessment for improved subjects

clinical_xl =  pd.read_excel('IN-TENS_Clinical_assessments_scores_all .xlsx', sheet_name=None)

subjects_part_ids = ['AmWo', 'FuMa', 'GrMa', 'KaBe', 'SoFa', 'WiLu', 'BuUl', 'EiHe', 'GuWi', 'MeRu']




fuma_score = np.zeros([10, 5]) 

for i, i_n in enumerate(subjects_part_ids):
    
    k = list(clinical_xl[str('Sheet1')][str('part_id')]).index(f'{i_n}')
    
    fuma_score[i, :] = clinical_xl[str('Sheet1')]['fmue'].iloc[np.arange(k, 95,19)]
    
    
fuma_score_swap = fuma_score[:, [0, 2, 1, 3 , 4]]
fuma_score_swap[5, 2] = 21 # just a given value    
time_points = ['v2', 'v3', 'v4', 'v5', 'v6']








improved_ind = np.array([0, 2, 3, 5, 6, 7, 8, 9])



p_t_im = {}
r_t_im = {}
analysis = ['amp', 'lat']

for i_analysis, analysis in enumerate(analysis):
    p_t_im[str(analysis)] = np.zeros([len(time_windows), len(time_points)])
    r_t_im[str(analysis)] = np.zeros([len(time_windows), len(time_points)])
    for i_components, n_components in enumerate(time_windows):         
        for i_time, n_time in enumerate(time_points):   
            if analysis == 'amp':
                r_t_im[str(analysis)][i_components, i_time], p_t_im[str(analysis)][i_components, i_time] = scipy.stats.pearsonr(fuma_score_swap[improved_ind, i_time], np.concatenate((amp_l_arr_st[i_components, np.array([0, 2, 3, 5]), i_time ], amp_r_arr_st[i_components, np.array([0, 1, 2, 3]), i_time])))
            elif analysis == 'lat':   
                r_t_im[str(analysis)][i_components, i_time], p_t_im[str(analysis)][i_components, i_time] = scipy.stats.pearsonr(fuma_score_swap[improved_ind, i_time], np.concatenate((lat_l_arr_st[i_components, np.array([0, 2, 3, 5]), i_time ], lat_r_arr_st[i_components, np.array([0, 1, 2, 3]), i_time])))
                
                
                
                


function_hc.FM_UE_plotting(amp_l_arr_st, amp_r_arr_st, fuma_score_swap, save_folder_peak)









#%%


p3_v2 = np.concatenate((amp_l_arr_st[5, np.array([0, 2, 3, 5]), 0], amp_r_arr_st[5, np.array([0, 1, 2, 3]), 0]))
p3_v6 = np.concatenate((amp_l_arr_st[5, np.array([0, 2, 3, 5]), 4], amp_r_arr_st[5, np.array([0, 1, 2, 3]), 4]))

n3_v2 = np.concatenate((amp_l_arr_st[4, np.array([0, 2, 3, 5]), 0], amp_r_arr_st[4, np.array([0, 1, 2, 3]), 0]))
n3_v6 = np.concatenate((amp_l_arr_st[4, np.array([0, 2, 3, 5]), 4], amp_r_arr_st[4, np.array([0, 1, 2, 3]), 4]))


n2_v2 = np.concatenate((amp_l_arr_st[2, np.array([0, 2, 3, 5]), 0], amp_r_arr_st[2, np.array([0, 1, 2, 3]), 0]))
n2_v6 = np.concatenate((amp_l_arr_st[2, np.array([0, 2, 3, 5]), 4], amp_r_arr_st[2, np.array([0, 1, 2, 3]), 4]))


FU_ME_v2 = fuma_score_swap[improved_ind, 0]
FU_ME_v6 = fuma_score_swap[improved_ind,4]


scipy.stats.pearsonr(p3_v2, FU_ME_v2) # V2
scipy.stats.pearsonr(p3_v6, FU_ME_v6) # V6
scipy.stats.pearsonr((p3_v2 + p3_v6), (FU_ME_v2 + FU_ME_v6)) # V2 + V6
scipy.stats.pearsonr((p3_v2 - p3_v6), (FU_ME_v2 - FU_ME_v6)) # V2 - V6
scipy.stats.pearsonr((p3_v6 - p3_v2), (FU_ME_v6 - FU_ME_v2)) # V6 - V2
scipy.stats.pearsonr((p3_v6 - p3_v2)/p3_v2, (FU_ME_v6 - FU_ME_v2)/FU_ME_v2) #  (V6 - V2)/ V2
scipy.stats.pearsonr((p3_v6 - p3_v2)/(p3_v6 + p3_v2) , (FU_ME_v6 - FU_ME_v2)/(FU_ME_v6 + FU_ME_v2)) # (V6 - V2)/(V6 + V2)


scipy.stats.pearsonr(n3_v2, FU_ME_v2) # V2
scipy.stats.pearsonr(n3_v6, FU_ME_v6) # V6
scipy.stats.pearsonr((n3_v2 + n3_v6), (FU_ME_v2 + FU_ME_v6)) # V2 + V6
scipy.stats.pearsonr((n3_v2 - n3_v6), (FU_ME_v2 - FU_ME_v6)) # V2 - V6
scipy.stats.pearsonr((n3_v6 - n3_v2), (FU_ME_v6 - FU_ME_v2)) # V6 - V2
scipy.stats.pearsonr((n3_v6 - n3_v2)/n3_v2, (FU_ME_v6 - FU_ME_v2)/FU_ME_v2) #  (V6 - V2)/ V2
scipy.stats.pearsonr((n3_v6 - n3_v2)/(n3_v6 + n3_v2) , (FU_ME_v6 - FU_ME_v2)/(FU_ME_v6 + FU_ME_v2)) # (V6 - V2)/(V6 + V2)



scipy.stats.pearsonr(n2_v2, FU_ME_v2) # V2
scipy.stats.pearsonr(n2_v6, FU_ME_v6) # V6
scipy.stats.pearsonr((n2_v2 + n2_v6), (FU_ME_v2 + FU_ME_v6)) # V2 + V6
scipy.stats.pearsonr((n2_v2 - n2_v6), (FU_ME_v2 - FU_ME_v6)) # V2 - V6
scipy.stats.pearsonr((n2_v6 - n2_v2), (FU_ME_v6 - FU_ME_v2)) # V6 - V2
scipy.stats.pearsonr((n2_v6 - n2_v2)/n2_v2, (FU_ME_v6 - FU_ME_v2)/FU_ME_v2) #  (V6 - V2)/ V2
scipy.stats.pearsonr((n2_v6 - n2_v2)/(n2_v6 + n2_v2) , (FU_ME_v6 - FU_ME_v2)/(FU_ME_v6 + FU_ME_v2)) # (V6 - V2)/(V6 + V2)




#%%

data = {'Subject ID' :  ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10'],
        'Lesion Side':  ['Left','Left','Left','Left','Left','Left', 'Right', 'Right', 'Right', 'Right'],
        'Paretic Side': ['Right', 'Right', 'Right', 'Right', 'Right', 'Right','Left','Left','Left','Left'], 
        'FM-UE (V2)'  : fuma_score_swap[:, 0],
        'FM-UE (V6)'  : fuma_score_swap[:, 4]
        }

df = pd.DataFrame(data)

fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center' )
for key, cell in table.get_celld().items():
    cell.set_edgecolor('grey')
    
# Color the header
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Header row
        cell.set_facecolor('lightgrey')  # Set your desired color here
plt.show()


#%%



#% Clinical Assessment for improved subjects

clinical_xl =  pd.read_excel('FMUE_best_all_hand.xlsx', sheet_name=None)

subjects_part_ids = ['AmWo', 'FuMa', 'GrMa', 'KaBe', 'SoFa', 'WiLu', 'BuUl', 'EiHe', 'GuWi', 'MeRu']




fuma_score_hand = np.array([[1, 1, 1, 1, 1], #AmWo
                            [1, 1, 1, 1, 1], #FuMa
                            [0, 0, 0, 0, 0], #GrMa
                            [5, 3, 4, 2, 6], #KaBe
                            [2, 1, 1, 1, 1], #SoFa
                            [2, 3, 3, 3, 2], #WiLu
                            [0, 1, 0, 0, 1], 
                            [1, 0, 0, 0, 0], 
                            [2, 3, 3, 5, 3], 
                            [1, 2, 3, 1, 1]]) 

time_points = ['v2', 'v3', 'v4', 'v5', 'v6']









import meta_functions_HC as function_hc


p_t = {}
r_t = {}
analysis = ['amp', 'lat']

for i_analysis, analysis in enumerate(analysis):
    p_t[str(analysis)] = np.zeros([len(time_windows), len(time_points)])
    r_t[str(analysis)] = np.zeros([len(time_windows), len(time_points)])
    for i_components, n_components in enumerate(time_windows):         
        for i_time, n_time in enumerate(time_points):   
            if analysis == 'amp':
                r_t[str(analysis)][i_components, i_time], p_t[str(analysis)][i_components, i_time] = scipy.stats.pearsonr(fuma_score_hand[:, i_time], np.concatenate((amp_l_arr_st[i_components,:, i_time ], amp_r_arr_st[i_components,:, i_time])))
            elif analysis == 'lat':   
                r_t[str(analysis)][i_components, i_time], p_t[str(analysis)][i_components, i_time] = scipy.stats.pearsonr(fuma_score_hand[:, i_time], np.concatenate((lat_l_arr_st[i_components,:, i_time ], lat_r_arr_st[i_components,:, i_time])))
                
                
                
                


function_hc.FM_UE_plotting_hand(amp_l_arr_st, amp_r_arr_st, fuma_score_hand, save_folder_peak)

#%%
import meta_functions_HC as function_hc
exdir_epoch_r = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/Right/v2/"
exdir_epoch_l = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/Left/v2/"

save_folder = '/home/sara/data/Third part/epochs_manually_rejected/V2/Group_models/com_ch_mai_8/'
win_erp0 = [15, 50]   
win_erp1 = [45, 85]   
win_erp2 = [100, 135] 
win_erp3 = [150, 220]
labels = ['P1', 'N1', 'P2', 'N2']
win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])

# HC group 
com_ind_l = {
 '0': ['C2', 'C4', 'CP2', 'CP4', 'CP6'],
 '1': ['FC1', 'FC2', 'FC4'],
 '2': ['F4','C4','T8','Fz','Cz','FC1','FC2','FC6','TP10','F2','C2','FC4','C6','FT8','TP8'],
 '3': ['C2', 'C4', 'FC4']}

com_ind_r = {
 '0': ['C3', 'C5', 'CP1', 'CP3', 'CP5', 'FT7', 'P1', 'P3', 'P5', 'T7'],
 '1': ['AF3', 'C1', 'Cz', 'F1', 'F2', 'F3', 'FC1', 'FC2', 'FC3', 'Fp1','Fp2', 'Fpz', 'Fz'],
 '2': ['T7', 'FC5', 'CP5', 'FT9', 'TP9', 'C5', 'FT7', 'TP7'],
 '3': ['CP2', 'CP4', 'P2', 'P4', 'P6', 'PO4', 'Pz']}






com_ind_l = {'0': function_hc.channel_indices(ch_com_l[str(0)]),  '1': function_hc.channel_indices(ch_com_l[str(1)]),  '2': function_hc.channel_indices(ch_names_l_hc[2]), '3': function_hc.channel_indices(ch_com_l[str(3)])}

com_ind_r = {'0': function_hc.channel_indices(ch_com_r[str(0)]),  '1': function_hc.channel_indices(ch_com_r[str(1)]), '2': function_hc.channel_indices(ch_names_r_hc[2]), '3': function_hc.channel_indices(ch_com_l[str(3)])}


labels = ['P1', 'N1', 'P2', 'N2']
win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])

# Group Model
phi_array_deg_correct_l_st, phi_array_deg_correct_r_st, mod_l_st, mod_r_st, p_r_st_g, p_l_st_g = function_hc.ST_fitting_cosine_plotting_left_right_group(win_erps, labels, com_ind_l, com_ind_r, exdir_epoch_l, exdir_epoch_r, save_folder)



# Plot modulation depth in bar format
amp_r = np.array(mod_r_st)
intensity_band = np.arange(2, 18, 2)
xi = range(len(np.array(intensity_band)))
fig = fig, ax = plt.subplots(1, 1)
            
plt.scatter(xi, amp_r[0, :],  c = 'maroon', label='P1')
plt.plot(xi, amp_r[0, :],  c = 'maroon', alpha=0.1)
plt.scatter(xi, amp_r[1, :],  c = 'navy', label='N1')
plt.plot(xi, amp_r[1, :],  c = 'navy', alpha=0.1)
plt.scatter(xi, amp_r[2, :],  c = 'coral', label='P2')
plt.plot(xi, amp_r[2, :],  c = 'coral', alpha=0.1)
plt.scatter(xi, amp_r[3, :],  c = 'b', label='N2')
plt.plot(xi, amp_r[3, :],  c = 'b', alpha=0.1)
plt.xlabel("Intensities (mA)", weight='bold')
plt.ylabel("Modulation depth", weight='bold')
plt.xticks(xi, np.array(intensity_band))
plt.legend(loc='upper right')
plt.ylim(0, 1)
plt.title(' Group Cosine Models', weight='bold')
plt.plot(7,  0.7, '*', c = 'coral')
#plt.plot(0,  0.7, '*', c = 'coral')
#plt.plot(7,  0.7, '*', c = 'navy')
plt.show()












# Individual Model
save_folder = '/home/sara/data/Third part/epochs_manually_rejected/V2/individual_model/com_ch_mai_8/'
ind_mod_l_st, ind_mod_r_st, ind_cosine_fit_l_st, ind_cosine_fit_r_st = function_hc.ST_fitting_cosine_plotting_left_right_ind(win_erps, labels, com_ind_l, com_ind_r, phi_array_deg_correct_l_st, phi_array_deg_correct_r_st, exdir_epoch_l, exdir_epoch_r, save_folder)



import meta_functions_HC as function_hc


win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])

# =============================================================================
# # HC group 
# com_ind_l = {
#     # C4, CP2, CP4, CP6, P2, P4, P6, PO4
#  '0': [5, 23, 43, 27, 37, 7, 51, 45],
#  '1': [35, 17], # C2, Cz
#  '2': [6, 7, 18, 37, 44, 45, 51, 62], # P3, P4, Pz, P2, PO3, PO4, P6, POz
#  '3': [3, 5, 13, 16, 17, 20, 21, 25, 31, 33, 35, 41, 47, 49, 55, 57]} 
# # F4, C4, T8, Fz, Cz, FC1, FC2, FC6, TP10, F2, C2, FC4, F6, C6, FT8, TP8
# 
# com_ind_r = {
#  '0': [42, 9, 63, 36, 6, 7, 50, 14, 15, 45, 59, 62, 18], #CP3, O2, Oz, P3, P4, P5, P7, P8, PO4, PO8, POz, Pz
#  '1': [38, 32, 33, 2, 20, 21, 40, 0, 16], # AF3, F1, F2, F3, FC1, FC2, FC3, Fp1, Fz
#  '2': [5, 6, 7, 9, 13, 15, 18, 19, 23, 27, 37, 43, 44, 45, 49, 51, 57, 59, 62, 63], # C4, P3, P4, O2, T8, P8, Pz, Iz, CP2, CP6, P2, CP4, PO3, PO4, C6, P6, TP8, PO8, POz, Oz
#  '3': [12, 24, 26, 28, 30, 40, 48, 54, 56]} # T7, FC5, CP5, FT9, TP9, FC3, C5, FT7, TP7
# 
# =============================================================================

exdir_epoch_r_hc = "/home/sara/data/Second part/epochs_manually_rejected/Right/"
exdir_epoch_l_hc = "/home/sara/data/Second part/epochs_manually_rejected/Left/"
save_folder =   '/home/sara/data/Second part/epochs_manually_rejected/Group_models/com_ch_mai_8/'
# Cosine fitting group HC
phi_array_deg_correct_l, phi_array_deg_correct_r,  amp_df_l, amp_df_r, p_r_hc_g, p_l_hc_g = function_hc.HC_fitting_cosine_plotting_left_right_group(win_erps, labels, com_ind_l, com_ind_r, exdir_epoch_l_hc, exdir_epoch_r_hc, save_folder)


# Plot modulation depth in bar format
amp_r = np.array(amp_df_r)
intensity_band = np.arange(2, 18, 2)
xi = range(len(np.array(intensity_band)))
fig = fig, ax = plt.subplots(1, 1)
            
plt.scatter(xi, amp_r[0, :],  c = 'maroon', label='P1')
plt.plot(xi, amp_r[0, :],  c = 'maroon', alpha=0.1)
plt.scatter(xi, amp_r[1, :],  c = 'navy', label='N1')
plt.plot(xi, amp_r[1, :],  c = 'navy', alpha=0.1)
plt.scatter(xi, amp_r[2, :],  c = 'coral', label='P2')
plt.plot(xi, amp_r[2, :],  c = 'coral', alpha=0.1)
plt.scatter(xi, amp_r[2, :],  c = 'b', label='N2')
plt.plot(xi, amp_r[2, :],  c = 'b', alpha=0.1)
plt.xlabel("Intensities (mA)", weight='bold')
plt.ylabel("Modulation depth", weight='bold')
plt.xticks(xi, np.array(intensity_band))
plt.legend(loc='upper right')
plt.ylim(0, 1)
plt.title(' Group Cosine Models', weight='bold')
plt.plot(7,  0.7, '*', c = 'maroon')
#plt.plot(0,  0.7, '*', c = 'coral')
#plt.plot(7,  0.7, '*', c = 'navy')
plt.show()


# Cosine fitting Individual HC
save_folder = '/home/sara/data/Second part/epochs_manually_rejected/Individual_model/com_ch_mai_8/'
mod_depth_l, mod_depth_r, ind_cosine_fit_l_hc, ind_cosine_fit_r_hc = function_hc.HC_fitting_cosine_plotting_left_right_ind(win_erps, labels, com_ind_l, com_ind_r, phi_array_deg_correct_l, phi_array_deg_correct_r, exdir_epoch_l_hc, exdir_epoch_r_hc, save_folder)

# =============================================================================
# int_band = np.arange(2, 18, 2)
# mod_depth =mod_depth_r
# amp_erp_all = [] 
# 
# for i_labels, name_labels in enumerate(labels):
#     for num_sub in range(8):  
#     
#         amp_erp_all.append(np.array(list(mod_depth[str(num_sub)][str(i_labels)].values())))
#     
#         
#     amp_erp_all_arr = np.array(amp_erp_all)
#  
#     
#     if  i_labels == 0 :
#         color = 'maroon'; component = 'P1'
#     elif i_labels == 1:
#         color = 'navy'; component = 'N1'
#     elif i_labels == 2:
#         color = 'coral'; component = 'P2'
#     
#     plt.plot(int_band, np.mean(amp_erp_all_arr, axis = 0 ), color = color, label = component, )
#     e_mod= np.std(amp_erp_all_arr, axis = 0 )
#     plt.errorbar(int_band, np.mean(amp_erp_all_arr, axis = 0 ), e_mod, color = color,  linestyle='None', marker='o' , alpha = 0.6 )
#     plt.title(' Individual Cosine Models', weight = 'bold')    
#     plt.ylabel('Modulation depth', weight = 'bold') 
#     plt.xlabel('Intensities (mA)', weight = 'bold')    
#     plt.legend(loc='upper right')
#     plt.ylim(bottom=0.1, top=1.5)
#     threshold = 4
#     plt.show()
#     plt.plot(16,  1.2, '*', c = 'maroon')
#     plt.plot(10,  1.2, '*', c = 'navy')
#     plt.plot(12,  1.2, '*', c = 'navy')
# 
# 
# =============================================================================

#%%

import meta_functions_HC as function_hc
save_folder_lme = '/home/sara/data/Third part/epochs_manually_rejected/V2_V6/R_results/'

time_windows = {'N1':[46, 80],'P1':[20, 50],'N2':[100, 135 ],'P2':[100, 135],'N3':[100, 135],'P3':[100, 135]}





    


amp_late = 'amp'
for i_time_windows, name_time_windows in enumerate(time_windows):
    function_hc.LME_mean_intensity_FuMe(i_time_windows, amp_late, name_time_windows, sub_names_L, sub_names_R, amp_l_arr_st, amp_r_arr_st, save_folder_lme)

    
amp_late = 'late'
for i_time_windows, name_time_windows in enumerate(time_windows):
    function_hc.LME_mean_intensity_FuMe(i_time_windows, amp_late, name_time_windows, sub_names_L, sub_names_R, lat_l_arr_st, lat_r_arr_st, save_folder_lme)
    
    
    
    
    
    