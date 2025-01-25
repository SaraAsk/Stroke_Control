#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:13:33 2024

@author: sara
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import meta_functions_HC as function_hc
#%%
exdir_epoch_r = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/Right/v2/"
exdir_epoch_l = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/Left/v2/"

win_erp0 = [15, 50]   
win_erp1 = [45, 85]   
win_erp2 = [100, 135] 
win_erp3 = [150, 220]
labels = ['P1', 'N1', 'N2', 'P2']
win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])

# HC group 
com_ch_l = {
 '0': ['C2', 'C4', 'CP2', 'CP4', 'CP6'],
 '1': ['FC1', 'FC2', 'FC4'],
 '2': ['F4','C4','T8','Fz','Cz','FC1','FC2','FC6','TP10','F2','C2','FC4','C6','FT8','TP8'],
 '3': ['C2', 'C4', 'FC4']}
com_ind_l = {
 '0': [35, 5, 23, 43, 27],
 '1': [20, 21, 41],
 '2': [3, 5, 13, 16, 17, 20, 21, 25, 31, 33, 35, 41, 49, 55, 57],
 '3': [35, 5, 41]}



com_ch_r = {
 '0': ['C3', 'C5', 'CP1', 'CP3', 'CP5', 'FT7', 'P1', 'P3', 'P5', 'T7'],
 '1': ['AF3', 'C1', 'Cz', 'F1', 'F2', 'F3', 'FC1', 'FC2', 'FC3', 'Fp1','Fp2', 'Fpz', 'Fz'],
 '2': ['T7', 'FC5', 'CP5', 'FT9', 'TP9', 'C5', 'FT7', 'TP7'],
 '3': ['CP2', 'CP4', 'P2', 'P4', 'P6', 'PO4', 'Pz']}
com_ind_r = {
 '0': [4, 48, 22, 42, 26, 54, 36, 6, 50, 12],
 '1': [38, 34, 17, 32, 33, 2, 20, 21, 40, 0, 1, 60, 16],
 '2': [12, 24, 26, 28, 30, 48, 54, 56],
 '3': [23, 43, 37, 7, 51, 45, 18]}

#%% Reading pickle files for stroke 

save_folder = '/home/sara/data/Third part/epochs_manually_rejected/V2/Group_models/com_ch_mai_8/'

        
with open(str(save_folder + '/right/') + 'cosinefit_r_st_group', 'rb') as fp:
         cosinefit_r_st = pickle.load(fp)

with open(str(save_folder + '/left/') + 'cosinefit_l_st_group', 'rb') as fp:
         cosinefit_l_st = pickle.load(fp)
         
         
    
         
with open(str(save_folder + '/right/') + 'ind_cosine_fit_r_st', 'rb') as fp:
         ind_cosine_fit_r_st = pickle.load(fp)
         
with open(str(save_folder + '/left/') + 'ind_cosine_fit_l_st', 'rb') as fp:
         ind_cosine_fit_l_st = pickle.load(fp)


#%%
import pickle

save_folder = '/home/sara/data/Third part/epochs_manually_rejected/V2/Group_models/com_ch_mai_8/'
# Group Model
#phi_array_deg_correct_l_st, phi_array_deg_correct_r_st, mod_l_st, mod_r_st, p_r_st_g, p_l_st_g = function_hc.ST_fitting_cosine_plotting_left_right_group(win_erps, labels, com_ind_l, com_ind_r, exdir_epoch_l, exdir_epoch_r, save_folder)


# Cosine fitting for the right hemisphere for each intensity stim_int = np.arange(2, 18, 2)
cosinefit_r_st, amplitudes_cosine_r_st, pvalues_cosine_r_st = function_hc.cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)
#     with open(str(save_folder_pickle) +pulses_ind_drop_filename, 'wb') as fp:
#         pickle.dump(pulses_ind_drop, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Further analysis just for the right hemisphere

# Circular correlation
phi_array_deg_r_st = function_hc.Circ_corr(cosinefit_r_st, labels)
# Modulation depth bar plot
p_val_r_st, phi_array_deg_correct_r_st, amp_df_r_st = function_hc.amp_p_chi(cosinefit_r_st, labels, phi_array_deg_r_st, save_folder + '/right/')
# Plotting best cosine fit and data
title = 'Group'
function_hc.best_fit_plot(cosinefit_r_st, labels, phi_array_deg_r_st, save_folder + '/right/' , title, [-1,1])

function_hc.fig_2c_scatter_sub_plot(cosinefit_r_st, labels, p_val_r_st, save_folder + '/right/')


# Cosine fitting for the left hemisphere for each intensity stim_int = np.arange(2, 18, 2)
cosinefit_l_st, amplitudes_cosine_l_st, pvalues_cosine_l_st = function_hc.cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
# Further analysis just for the right hemisphere

# Circular correlation
phi_array_deg_l_st = function_hc.Circ_corr(cosinefit_l_st, labels)
# Modulation depth bar plot
p_val_l_st, phi_array_deg_correct_l_st, amp_df_l_st = function_hc.amp_p_chi(cosinefit_l_st, labels, phi_array_deg_l_st, save_folder + '/left/')
# Plotting best cosine fit and data
title = 'Group'
function_hc.best_fit_plot(cosinefit_l_st, labels, phi_array_deg_l_st, save_folder + 'left/', title, [-1,1])
function_hc.fig_2c_scatter_sub_plot(cosinefit_l_st, labels, p_val_l_st, save_folder + 'left/')


cosinefit_r_avg_int_st, amplitudes_cosine_r_avg_int_st, pvalues_cosine_r_avg_int_st, surrogate_r_avg_st  = function_hc.cosine_fit_phase_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)
cosinefit_l_avg_int_st, amplitudes_cosine_l_avg_int_st, pvalues_cosine_l_avg_int_st, surrogate_l_avg_st  = function_hc.cosine_fit_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
function_hc.bar_plot_avg_intensity(labels, amplitudes_cosine_r_avg_int_st, surrogate_r_avg_st, pvalues_cosine_r_avg_int_st, 'Group Cosine Model for avg Intensities' , save_folder + 'right/')
function_hc.bar_plot_avg_intensity(labels, amplitudes_cosine_l_avg_int_st, surrogate_l_avg_st, pvalues_cosine_l_avg_int_st, 'Group Cosine Model for avg Intensities' , save_folder + 'left/')


####################################################################################################################


# Individual Model
#ind_mod_l_st, ind_mod_r_st, ind_cosine_fit_l_st, ind_cosine_fit_r_st = function_hc.ST_fitting_cosine_plotting_left_right_ind(win_erps, labels, com_ind_l, com_ind_r, phi_array_deg_correct_l_st, phi_array_deg_correct_r_st, exdir_epoch_l, exdir_epoch_r, save_folder)

ind_amplitudes_cosine_l_st, ind_cosine_fit_l_st ,ind_subjects_names_l_st = function_hc.individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
mod_depth_ind_l_st, surrogate_ind_l_st, phi_ind_l_st = function_hc.reading_cosine_function_parameters(ind_cosine_fit_l_st, labels) 
function_hc.subplot_torrecillos_2c_errorbar(mod_depth_ind_l_st, surrogate_ind_l_st, labels, save_folder + 'left/')
bin_class_all_ind_l_st, phi_tar_freq_all_ind_l_st =  function_hc.phase_to_bin_class(ind_cosine_fit_l_st, phi_ind_l_st, labels)   
plt.style.use('default')  
titles = ['2 mA', '4 mA', '6 mA', '8 mA', '10 mA', '12 mA', '14 mA', '16 mA']   
fig = plt.figure(constrained_layout=True, figsize=(20,12))
#fig.suptitle('Optimal Phase distribution', fontweight="bold")
# create 3x1 subfigs
subfigs = fig.subfigures(nrows=len(labels), ncols=1)
theta = {}
theta_g = {}
for row, subfig in enumerate(subfigs):
    
    axs = subfig.subplots(nrows=1, ncols=8,subplot_kw=dict(projection='polar'))
    for intensity, ax in enumerate(axs):
         print(intensity, ax)
         axs[0].set_ylabel(f'{labels[row]}', rotation=0, size=14, labelpad = 50, fontweight="bold")
         theta[str(intensity)], theta_g[str(intensity)] = function_hc.phase_optimal_per_sub(ax,  bin_class_all_ind_l_st[str(row)][intensity,:],  phi_tar_freq_all_ind_l_st[str(row)][intensity,:], phi_array_deg_correct_l_st[intensity,row], titles[intensity])   
fig.savefig(save_folder  + 'left/' +'optimal_phase_distribution.svg')    


###############################################################################
# Right
ind_amplitudes_cosine_r_st, ind_cosine_fit_r_st ,ind_subjects_names_r_st = function_hc.individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)
mod_depth_ind_r_st, surrogate_ind_r_st, phi_ind_r_st = function_hc.reading_cosine_function_parameters(ind_cosine_fit_r_st, labels) 
function_hc.subplot_torrecillos_2c_errorbar(mod_depth_ind_r_st, surrogate_ind_r_st, labels, save_folder + 'right/')
bin_class_all_ind_r_st, phi_tar_freq_all_ind_r_st =function_hc.phase_to_bin_class(ind_cosine_fit_r_st, phi_ind_r_st, labels)  
plt.style.use('default')  
titles = ['2 mA', '4 mA', '6 mA', '8 mA', '10 mA', '12 mA', '14 mA', '16 mA']   
fig = plt.figure(constrained_layout=True, figsize=(20,12))
#fig.suptitle('Optimal Phase distribution', fontweight="bold")
# create 3x1 subfigs
subfigs = fig.subfigures(nrows=len(labels), ncols=1)
theta = {}
theta_g = {}
for row, subfig in enumerate(subfigs):
    
    axs = subfig.subplots(nrows=1, ncols=8,subplot_kw=dict(projection='polar'))
    for intensity, ax in enumerate(axs):
         axs[0].set_ylabel(f'{labels[row]}', rotation=0, size=14, labelpad = 50, fontweight="bold")

         theta[str(intensity)], theta_g[str(intensity)] = function_hc.phase_optimal_per_sub(ax,  bin_class_all_ind_r_st[str(row)][intensity,:],  phi_tar_freq_all_ind_r_st[str(row)][intensity,:], phi_array_deg_correct_r_st[intensity,row], titles[intensity])   
 
fig.savefig(save_folder  + 'right/' +'optimal_phase_distribution.svg')   








# Individual avg intensity
#avg_ind_amplitudes_cosine_l_st, avg_ind_cosine_fit_l_st   = function_hc.cosine_fit_phase_ind_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
# avg_ind_amplitudes_cosine_r_st, avg_ind_cosine_fit_r_st   = function_hc.cosine_fit_phase_ind_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)



#avg_mod_depth_ind_r_st, avg_surrogate_ind_r_st, avg_phi_ind_r_st = function_hc.reading_cosine_function_parameters_just_phase(avg_ind_amplitudes_cosine_r_st, labels) 
#avg_mod_depth_ind_l_st, avg_surrogate_ind_l_st, avg_phi_ind_l_st = function_hc.reading_cosine_function_parameters_just_phase(avg_ind_amplitudes_cosine_l_st, labels) 
#function_hc.bar_plot_avg_intensity_ind(labels, avg_mod_depth_ind_l_st, avg_surrogate_ind_l_st, avg_phi_ind_l_st, 'Individual Cosine Model for avg Intensities' , save_folder + 'right/')
#function_hc.bar_plot_avg_intensity_ind(labels, avg_mod_depth_ind_r_st, avg_surrogate_ind_r_st, avg_phi_ind_r_st, 'Individual Cosine Model for avg Intensities' , save_folder + 'right/')






with open(str(save_folder + '/right/') + 'cosinefit_r_st_group', 'wb') as fp:
         pickle.dump(cosinefit_r_st, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(str(save_folder + '/left/') + 'cosinefit_l_st_group', 'wb') as fp:
         pickle.dump(cosinefit_l_st, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(str(save_folder + '/right/') + 'ind_cosine_fit_r_st', 'wb') as fp:
         pickle.dump(ind_cosine_fit_r_st, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(str(save_folder + '/left/') + 'ind_cosine_fit_l_st', 'wb') as fp:
         pickle.dump(ind_cosine_fit_l_st, fp, protocol=pickle.HIGHEST_PROTOCOL)
         
         
         
         
#%% Reading pickle files for Control 

save_folder =  '/home/sara/data/Second part/epochs_manually_rejected/Individual_model/com_ch_mai_8'

        
with open(str(save_folder + '/right/') + 'cosinefit_r_hc_group', 'rb') as fp:
         cosinefit_r = pickle.load(fp)

with open(str(save_folder + '/left/') + 'cosinefit_l_hc_group', 'rb') as fp:
         cosinefit_l = pickle.load(fp)
         
         
    
         
with open(str(save_folder + '/right/') + 'ind_cosine_fit_r_hc', 'rb') as fp:
         ind_cosine_fit_r = pickle.load(fp)
         
with open(str(save_folder + '/left/') + 'ind_cosine_fit_l_hc', 'rb') as fp:
         ind_cosine_fit_l = pickle.load(fp)
         









         
         
exdir_epoch_r_hc = "/home/sara/data/Second part/epochs_manually_rejected/Right/"
exdir_epoch_l_hc = "/home/sara/data/Second part/epochs_manually_rejected/Left/"
save_folder =   '/home/sara/data/Second part/epochs_manually_rejected/Group_models/com_ch_mai_8/'
# Cosine fitting group HC
#phi_array_deg_correct_l, phi_array_deg_correct_r,  amp_df_l, amp_df_r, p_r_hc_g, p_l_hc_g = function_hc.HC_fitting_cosine_plotting_left_right_group(win_erps, labels, com_ind_l, com_ind_r, exdir_epoch_l_hc, exdir_epoch_r_hc, save_folder)


# Cosine fitting for the right hemisphere for each intensity stim_int = np.arange(2, 18, 2)
cosinefit_r, amplitudes_cosine_r, pvalues_cosine_r = function_hc.cosine_fit_intensity_phase_overlapping_chs( win_erps, labels, list(com_ind_r.values()), exdir_epoch_r_hc)
# Further analysis just for the right hemisphere
# Circular correlation
phi_array_deg_r = function_hc.Circ_corr(cosinefit_r, labels)
# Modulation depth bar plot
p_val_r, phi_array_deg_correct_r, amp_df_r = function_hc.amp_p_chi(cosinefit_r, labels, phi_array_deg_r, save_folder + 'com_ch/right/')
# Plotting best cosine fit and data
title = 'Group'
function_hc.best_fit_plot(cosinefit_r, labels, phi_array_deg_r, save_folder + 'com_ch/right/' , title, [-1,1])
function_hc.fig_2c_scatter_sub_plot(cosinefit_r, labels, p_val_r, save_folder + 'com_ch/right/')
# Cosine fitting for the left hemisphere for each intensity stim_int = np.arange(2, 18, 2)
cosinefit_l, amplitudes_cosine_l, pvalues_cosine_l = function_hc.cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l_hc)
# Further analysis just for the right hemisphere
# Circular correlation
phi_array_deg_l = function_hc.Circ_corr(cosinefit_l, labels)
# Modulation depth bar plot
p_val_l, phi_array_deg_correct_l, amp_df_l = function_hc.amp_p_chi(cosinefit_l, labels, phi_array_deg_l, save_folder + 'com_ch/left/')
# Plotting best cosine fit and data
title = 'Group'
function_hc.best_fit_plot(cosinefit_l, labels, phi_array_deg_l, save_folder + 'com_ch/left/', title, [-1,1])
function_hc.fig_2c_scatter_sub_plot(cosinefit_l, labels, p_val_l, save_folder + 'com_ch/left/')
cosinefit_r_avg_int, amplitudes_cosine_r_avg_int, pvalues_cosine_r_avg_int, surrogate_r_avg  = function_hc.cosine_fit_phase_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r_hc)
cosinefit_l_avg_int, amplitudes_cosine_l_avg_int, pvalues_cosine_l_avg_int, surrogate_l_avg  = function_hc.cosine_fit_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l_hc)
function_hc.bar_plot_avg_intensity(labels, amplitudes_cosine_r_avg_int, surrogate_r_avg, pvalues_cosine_r_avg_int, 'Group Cosine Model for avg Intensities' , save_folder + 'com_ch/right/')
function_hc.bar_plot_avg_intensity(labels, amplitudes_cosine_l_avg_int, surrogate_l_avg, pvalues_cosine_l_avg_int, 'Group Cosine Model for avg Intensities' , save_folder + 'com_ch/left/')



###########################################################################################################################

# Cosine fitting Individual HC
save_folder = '/home/sara/data/Second part/epochs_manually_rejected/Individual_model/com_ch_mai_8/'
#mod_depth_l, mod_depth_r, ind_cosine_fit_l_hc, ind_cosine_fit_r_hc = function_hc.HC_fitting_cosine_plotting_left_right_ind(win_erps, labels, com_ind_l, com_ind_r, phi_array_deg_correct_l, phi_array_deg_correct_r, exdir_epoch_l_hc, exdir_epoch_r_hc, save_folder)

####################################################################################
# Left
ind_amplitudes_cosine_l, ind_cosine_fit_l ,ind_subjects_names_l = function_hc.individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l_hc)
mod_depth_l, surrogate_l, phi = function_hc.reading_cosine_function_parameters(ind_cosine_fit_l, labels) 
function_hc.subplot_torrecillos_2c_errorbar(mod_depth_l, surrogate_l, labels, save_folder + 'left/')
bin_class_all, phi_tar_freq_all =  function_hc.phase_to_bin_class(ind_cosine_fit_l, phi, labels)
    
plt.style.use('default')  
titles = ['2 mA', '4 mA', '6 mA', '8 mA', '10 mA', '12 mA', '14 mA', '16 mA']   
fig = plt.figure(constrained_layout=True, figsize=(20,12))
#fig.suptitle('Optimal Phase distribution', fontweight="bold")
# create 3x1 subfigs
subfigs = fig.subfigures(nrows=len(labels), ncols=1)
theta = {}
theta_g = {}
for row, subfig in enumerate(subfigs):
    
    axs = subfig.subplots(nrows=1, ncols=8,subplot_kw=dict(projection='polar'))
    for intensity, ax in enumerate(axs):
         axs[0].set_ylabel(f'{labels[row]}', rotation=0, size=14, labelpad = 50, fontweight="bold")
         theta[str(intensity)], theta_g[str(intensity)] = function_hc.phase_optimal_per_sub(ax,  bin_class_all[str(row)][intensity,:],  phi_tar_freq_all[str(row)][intensity,:], phi_array_deg_correct_l[intensity,row], titles[intensity])   
 
fig.savefig(save_folder  + 'left/' +'optimal_phase_distribution.svg')         
   


####################################################################################
# Right
ind_amplitudes_cosine_r, ind_cosine_fit_r ,ind_subjects_names_r = function_hc.individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_r_hc)
mod_depth_r, surrogate_r, phi = function_hc.reading_cosine_function_parameters(ind_cosine_fit_r, labels) 
function_hc.subplot_torrecillos_2c_errorbar(mod_depth_r, surrogate_r, labels, save_folder + 'right/')




bin_class_all, phi_tar_freq_all =  function_hc.phase_to_bin_class(ind_cosine_fit_r, phi, labels)
    
plt.style.use('default')  
titles = ['2 mA', '4 mA', '6 mA', '8 mA', '10 mA', '12 mA', '14 mA', '16 mA']   
fig = plt.figure(constrained_layout=True, figsize=(20,12))
#fig.suptitle('Optimal Phase distribution', fontweight="bold")
# create 3x1 subfigs
subfigs = fig.subfigures(nrows=len(labels), ncols=1)
theta = {}
theta_g = {}
for row, subfig in enumerate(subfigs):
    
    axs = subfig.subplots(nrows=1, ncols=8,subplot_kw=dict(projection='polar'))
    for intensity, ax in enumerate(axs):
         axs[0].set_ylabel(f'{labels[row]}', rotation=0, size=14, labelpad = 50, fontweight="bold")
         theta[str(intensity)], theta_g[str(intensity)] = function_hc.phase_optimal_per_sub(ax,  bin_class_all[str(row)][intensity,:],  phi_tar_freq_all[str(row)][intensity,:], phi_array_deg_correct_r[intensity,row], titles[intensity])   
 
fig.savefig(save_folder  + 'right/' +'optimal_phase_distribution.svg')         



with open(str(save_folder + '/right/') + 'cosinefit_r_hc_group', 'wb') as fp:
         pickle.dump(cosinefit_r, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(str(save_folder + '/left/') + 'cosinefit_l_hc_group', 'wb') as fp:
         pickle.dump(cosinefit_l, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(str(save_folder + '/right/') + 'ind_cosine_fit_r_hc', 'wb') as fp:
         pickle.dump(ind_amplitudes_cosine_r, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(str(save_folder + '/left/') + 'ind_cosine_fit_l_hc', 'wb') as fp:
         pickle.dump(ind_amplitudes_cosine_l, fp, protocol=pickle.HIGHEST_PROTOCOL)













