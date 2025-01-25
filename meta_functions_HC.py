#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:42:57 2022

@author: sara
"""

import mne
import math
import json
import pyxdf
import pycircstat
import numpy as np
from scipy import stats
#import pickle
import lmfit
import itertools  
import mne.stats
from tqdm import tqdm
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.stats import  zscore
from mne.channels import make_standard_montage
import matplotlib.pyplot as plt 
import pandas as pd
from mne.stats import permutation_cluster_test
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from scipy.signal import medfilt
import scipy.stats 
#import meta_functions_hc as function_hc
import pickle
# =============================================================================



def XDF_to_epochs_Hc(exdir, save_folder, save_folder_figs):
    files = list(Path(exdir).glob('*.xdf*'))

    
    
    
    
    # Information here are provided based on healthy controls' protocol and reiz_marker_sa labels
    # e.g. based on the protocol we know which hand was stimulated first, based on the reizmarker's labels we know if all the 8 phases and I/O stimuli are saved in the same file
    dict_origin_labels  = {'TMS_NMES_AmKa_healthy.xdf':['R', 'L'], 
                           'TMS_NMES_GuCh_healthy.xdf':['R', 'L'],
                           'TMS_NMES_KoCh_healthy.xdf':['R', 'L'], 
                           'TMS_NMES_KoEb_healthy_old1.xdf': ['L', 'R'],
                           'TMS_NMES_RaPa_healthy_old1.xdf': ['L'],
                           'TMS_NMES_RaPa_healthy_old2.xdf': ['L', 'R'], 
                           'TMS_NMES_RaSi_healthy.xdf': ['L', 'R'],
                           'TMS_NMES_SoMa_healthy_old1.xdf': ['R', 'L'],
                           'TMS_NMES_TrGe_healthy.xdf':['L', 'R'],
                           'TMS_NMES_VoAn_healthy.xdf':['L', 'R'],
                           'TMS_NMES_VoHe_healthy.xdf':['L', 'R'],
                           'TMS_NMES_VoSa_healthy.xdf':['R', 'L'],
                           'TMS_NMES_WiBe_healthy.xdf':['R', 'L'],
                           'TMS_NMES_WiDo_healthy.xdf':['R', 'L'],
                           'TMS_NMES_ZaBe_healthy_old1.xdf':['R', 'L']
                           }
    
                                                                                       #/\      
    # subjects with huge stimulation artifact at the time of trigger (like TMS --------/  \--------)
    stim_artifact_subs = { 'AmKa': 'R','KoEb' : 'R', 'VoAn' : 'R', 'VoSa': 'R'}
    
        
        
        
    
    #lists_bad_channels = []
    lists_pluses = []
    for f in files:
        plt.close('all')
        all_possible_stim_site = dict_origin_labels[f.parts[-1]]
        print(str(f.parts[5]) + '_' + str(f.parts[-1][39:43])) # print sb's name
        marker_n = pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
        brainvision = pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
        #C3 = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == 'C3'][0]
        #C3dat = brainvision['time_series'][:,C3]
        out = {'pulse_BV':[], 'drop_idx_list': [], 'pulse_BV_idx_list':[]}
        # bipolar signal near to C3
        #bipolar = pyxdf.load_xdf(f, select_streams=[{'name': 'Spongebob-Data'}])[0][0]
        # pulses creates a list of the indices of the marker timestamps for the stimulation condition trials only
        # recognizing the pulses by checking if the marker value starts with a number 
        pulses = [i for i,m in enumerate(marker_n['time_series']) if marker_n['time_series'][i][0][1].isnumeric()]
        
    
        
    
        
        pulses_8_phases = [i for i,m in enumerate(marker_n['time_series']) if "\"phases_to_stimulate\": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]" in m[0]]
        # deleting the indexes with repeated stimulation marker with a distance of 1, e.g. AmKa marker indexs 11 and 12
        if (np.diff(pulses_8_phases) ==1).any() ==True: 
            pulses_8_phases = np.delete(pulses_8_phases, np.where(np.diff(pulses_8_phases) ==1)[0][0]) # staring point
        # extra condition for 'Wibe', since there are extra trials for 8 phases
        if f.parts[-1][9:13] == 'WiBe':
            pulses_8_phases = pulses_8_phases[1:]
        
            # extra condition for 'Wibe', since there are extra trials for 8 phases
        if f.parts[-1][9:13] == 'KoEb':
            pulses_8_phases = pulses_8_phases[1:]
        
        # extra condition for 'VoAn', since BrainVision time series start with intensity and phase label
        if f.parts[-1][9:13] == 'VoAn':
            pulses_8_phases = [0, pulses_8_phases[0]]
            
        if f.parts[-1][9:13] == 'SoMa':
            pulses_8_phases = pulses_8_phases[1:]
            
        if f.parts[-1][9:13] == 'VoSa':
            pulses_8_phases = pulses_8_phases[2:]
    
    
        pulses_0 = [i for i,m in enumerate(marker_n['time_series']) if "\"phases_to_stimulate\": [0.0]" in m[0]]
        if (np.diff(pulses_0) ==1).any() ==True:
            pulses_0 = np.delete(pulses_0, np.where(np.diff(pulses_0) ==1)[0][0]) # staring point
        else:
            pulses_0 = pulses_0    
               
            
        pulses_180 = [i for i,m in enumerate(marker_n['time_series']) if "\"phases_to_stimulate\": [180.0]" in m[0]]
        if (np.diff(pulses_180) ==1).any() ==True:
            pulses_180 = np.delete(pulses_180, np.where(np.diff(pulses_180) ==1)[0][0]) # staring point
        else:
            pulses_180 = pulses_180        
            
            
            
            
        pulses = [i for i,m in enumerate(marker_n['time_series']) if marker_n['time_series'][i][0][1].isnumeric()and i >= pulses_8_phases[0]]    
            
        # pulseinfo contains a list of the stim.condition time stamps and descriptions
        # each item in the list contains a list with the size 2: pulseinfo[i][0] is the timestamp corresponding with the index i from pulses,
        # pulseinfo[i][1] contains the corresponding stimulus description (i.e., stim phase and freq, etc.)
        pulseinfo = [[np.searchsorted(brainvision['time_stamps'], marker_n['time_stamps'][p]), marker_n['time_series'][p]] for p in pulses]

        

        # Here we need to set EDC muscle of the right or left hand according to the stimulation site    
        for i, v in enumerate(all_possible_stim_site):
     
            if v == 'R':
                EDC_hand = 'EDC_R'
                pulseinfo = pulseinfo
            elif v == 'L':
                EDC_hand = 'EDC_L'
                pulseinfo = pulseinfo
            print(v, EDC_hand)
        
        
            edcix = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == EDC_hand][0]
            edcdat = brainvision['time_series'][:,edcix]
            print(v, EDC_hand, edcix)
            plt.plot(edcdat)
        
    
            
        
            for i,p in enumerate(pulseinfo):
                
                pulse_idx = pulses[pulseinfo.index(p)]
                sample = p[0]
        
                # For the NMES study, we use the ECD_R data to identify the artifact
                # and we use a time window around the onset of the original reizmarker_timestamp: [sample-1500:sample+1500]
                if sample > 1500:
                    onset = sample-1500
                    offset = sample+1500
                    edcep = edcdat[onset:offset]
                    #C3ep = C3dat[onset:offset]
                    dmy =  np.abs(zscore(edcep))
                    #dmy_c3  = np.abs(zscore(edcep))
                    tartifact = np.argmax(dmy)
                    #tartifact_v = max(dmy) 
                    #peaks, _ = find_peaks(dmy, height=3, distance = 50)
                    
                   
                    

                        # if there is just one stimulated artifact around onset and offset of the reizmarker_timestamp
                        #if len(peaks) == 1 :
                            
                    corrected_timestamp = sample - 1500 + tartifact
                    out['pulse_BV'].append(corrected_timestamp)
                    out['pulse_BV_idx_list'].append(pulse_idx)
                            # =============================================================================
                            #                     # This part is to select time of trigger manually                   
                            #                     fig, ax = plt.subplots( figsize=(15, 15))
                            #                     ax.plot(C3dat[onset:offset])
                            #                     ax.plot(edcdat[onset:offset]/100)
                            #                     klicker = clicker(ax, ["event"], markers=["x"])
                            #                     clicks = fig.ginput(n=1, timeout=4)
                            #                     plt.show()
                            #                     if len(klicker.get_positions()['event'])> 0:
                            #                         cordinate_val = int(klicker.get_positions()['event'][0][0])
                            #                         a.append(sample - 1500 + cordinate_val)
                            #                     plt.close()
                            # =============================================================================
                
# =============================================================================
#                         elif  (len(peaks)==2 and np.diff(peaks)>1000): 
#                             # if there is 2 stimulated artifacts around the onset and offset of reizmarker_timestamp, the choose the ones with 1000 distance. I chose this number by try and error based on how many trials will be rejected in the ned as well as how ERPs will look like. 
#                             
#                             # edcep contains 3000 timepoints or samples (-1500 to +1500 samples around the original rm_marker)
#                             # so, if tartifact is < 1500, the new marker is in the period before the original marker
#                             # if tartifact is >1500, the new marker is in the period after the original marker      
#                             corrected_timestamp = sample - 1500 + tartifact
#                             out['pulse_BV'].append(corrected_timestamp)
#                             out['pulse_BV_idx_list'].append(pulse_idx)
# =============================================================================
                        
                        #samp_stamp_diff.append(corrected_timestamp-sample)
                        #print(np.argmax(dmy), np.max(dmy), tartifact, tartifact_v)
                            # the section below is to check for trials where no clear stimulation artifact is present
                # a list of indices is created and saved in out['drop_idx_list'], to be used to reject 
                # these epochs when the preprocessing in MNE is started

                else:
                    # VoAn subject has a sample = 1004, so to avoid the error
                    onset = sample-1000
                    offset = sample+1000
                    edcep = edcdat[onset:offset]
                    dmy= np.abs(zscore(edcep))
                    #dmy_c3  = np.abs(zscore(edcep))
                    tartifact = np.argmax(dmy)
                    #tartifact_v = max(dmy) 
                    corrected_timestamp = sample - 1000 + tartifact
                    out['pulse_BV_idx_list'].append(pulse_idx)

        

    
            if np.max(dmy) < 3 :
               
                out['drop_idx_list'].append(pulse_idx)
            out['pulse_BV'].append(corrected_timestamp)
            _, _, pulses_ind_drop = np.intersect1d(out['drop_idx_list'], pulses, return_indices=True)
        marker_corrected = marker_n
            
# =============================================================================
#             for i in range(len(out['pulse_BV_idx_list'])):
#                 # for the stim.condition time stamps (corresponding to the indices stored in pulses)
#                 # replace original reizmarker (rm) timestamp value with the corrected timestamp value based on the EDC artifact (corrected_timestamp)
#                 rm_timestamp_idx = out['pulse_BV_idx_list'][i]
#                 if i< len(out['pulse_BV']):
#                     brainvision_idx = out['pulse_BV'][i]
#                     rm_timestamp_new_value = brainvision['time_stamps'][brainvision_idx] 
#                             
#                     #print('old value: '+str(marker['time_stamps'][pulses[i]]))
#                     # replace original stimulus onset time stamp with the new timestamp value
#                     marker_corrected['time_stamps'][rm_timestamp_idx] = rm_timestamp_new_value
#                     #print('new value: '+str(marker['time_stamps'][pulses[i]]))
# =============================================================================
            
        for i in range(len(pulses)):
            # for the stim.condition time stamps (corresponding to the indices stored in pulses)
            # replace original reizmarker (rm) timestamp value with the corrected timestamp value based on the EDC artifact (corrected_timestamp)
            rm_timestamp_idx = pulses[i]
            brainvision_idx = out['pulse_BV'][i]
            rm_timestamp_new_value = brainvision['time_stamps'][brainvision_idx] 
                    
            #print('old value: '+str(marker['time_stamps'][pulses[i]]))
            # replace original stimulus onset time stamp with the new timestamp value
            marker_corrected['time_stamps'][rm_timestamp_idx] = rm_timestamp_new_value
            #print('new value: '+str(marker['time_stamps'][pulses[i]]))    
        
            #### convert brainvision and corrected marker stream into a fif file that can be read by MNE ###    
        
        #marker_corrected = marker    #pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
        data = brainvision   #pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
        marker_corrected['time_stamps'] -= data['time_stamps'][0] #remove clock offset
        
        channel_names = [c['label'][0] for c in data['info']['desc'][0]['channels'][0]['channel'] ]
        sfreq = int(data['info']['nominal_srate'][0])
        types = ['eeg']*64
        types.extend(['emg']*(len(channel_names)-64)) #64 EEG chans, rest is EMG/EKG
        info = mne.create_info(ch_names = channel_names, sfreq = sfreq, ch_types = types)
        #raw = mne.io.RawArray(data = data['time_series'][0:int(marker_corrected['time_stamps'][-1]*1500), :].T, info = info)
        raw = mne.io.RawArray(data = data['time_series'].T,info = info)
        if len(marker_corrected['time_stamps']) > 1:
            descs = [msg[0] for msg in marker_corrected['time_series']]
            ts = marker_corrected['time_stamps']
            
            sel = [i for i,v in enumerate(descs)  if marker_n['time_series'][i][0][1].isnumeric() and i >= pulses_8_phases[0]]
            if f.parts[-1] == 'TMS_NMES_RaPa_healthy_old2.xdf':
                  sel = [i for i,v in enumerate(descs)  if marker_n['time_series'][i][0][1].isnumeric() ]
                
    
            # sel values start from the pulses with numeric values
            #descs = [descs[i] for i in sel]
            
            if f.parts[-1] == 'TMS_NMES_RaPa_healthy_old1.xdf':
                for i in sel:
                    # add "L" or "R" to the labels according to healthy control protocol
                    if  i in range(pulses_8_phases[0]+1, sel[-1]+1, 1):
                        descs[i] = dict_origin_labels[f.parts[-1]][0]+ '_' +  descs[i]
                        
            elif f.parts[-1] == 'TMS_NMES_RaPa_healthy_old2.xdf':
                for i in sel:
                    # add "IO" for extra  phases 0 and 180
                    if  i in range(pulses_0[0], pulses_8_phases[0], 1):
                        descs[i] = dict_origin_labels[f.parts[-1]][0] + '_' + 'IO' + '_' +  descs[i]             
                    if  i in range(pulses_8_phases[0]+1, pulses_0[1], 1):
                        descs[i] = dict_origin_labels[f.parts[-1]][1]   + '_' +  descs[i]       
                    if  i in range(pulses_0[1]+1, sel[-1]+1, 1):
                        descs[i] = dict_origin_labels[f.parts[-1]][1]  + '_'  +'IO' + '_' +  descs[i]
                
            else:            
                for i_sel, v_sel in enumerate(sel):
                    # add "L" or "R" to the labels according to healthy control protocol
                    if  v_sel in range(pulses_8_phases[0]+1, pulses_0[0], 1):
                        descs[v_sel] = dict_origin_labels[f.parts[-1]][0] + '_' +  descs[v_sel]
                    # add "IO" for extra  phases 0 and 180
                    if  v_sel in range(pulses_0[0]+1, pulses_8_phases[1], 1):
                        descs[v_sel] = dict_origin_labels[f.parts[-1]][0] + '_'  + 'IO'  +  '_' +  descs[v_sel]
                        
                    if  v_sel in range(pulses_8_phases[1]+1, pulses_0[1], 1):
                        descs[v_sel] = dict_origin_labels[f.parts[-1]][1]    + '_' +  descs[v_sel]
                        
                    if  v_sel in range(pulses_0[1]+1, sel[-1]+1, 1):
                        descs[v_sel] = dict_origin_labels[f.parts[-1]][1] + '_' +'IO' +  '_' +  descs[v_sel]
                
    
            
        descs = [descs[i] for i in sel]        
        ts = [ts[i] for i in sel]

    ts_new = np.delete(ts, pulses_ind_drop)
    shortdescs_new = np.delete(descs, pulses_ind_drop)
    
    
    
    
    
      
    # Find the latency when stimulation switches from one side to the other side
    switch_ind = [i for i,v in enumerate(shortdescs_new)  if v[0]!= dict_origin_labels[f.parts[-1]][0] and v[0]!="'"]

        
   
    for i, v in enumerate(all_possible_stim_site):
        
        if i==0 and len(all_possible_stim_site)>1:
            ts_switch= ts_new[0:switch_ind[0]]
            shortdescs_switch = shortdescs_new[0:switch_ind[0]]
        elif i==1 and len(all_possible_stim_site)>1:    
            ts_switch = ts_new[switch_ind[0]:switch_ind[-1]+1] 
            shortdescs_switch = shortdescs_new[switch_ind[0]:switch_ind[-1]+1]  
        else:
            # To compensate for the cutted file of RaPa subject, TMS_NMES_RaPa_healthy_old1. Just left side is stimulated
             shortdescs_switch =  shortdescs_new 
             ts_switch = ts_new

    
    
    
        anno = mne.Annotations(onset = ts_switch, duration = 0, description = shortdescs_switch)
        raw = raw.set_annotations(anno)  
        if i==0:
            raw.pick_channels(['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','Iz','FC1','FC2','CP1',
                                'CP2','FC5','FC6','CP5','CP6','FT9','FT10','TP9','TP10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4','CP3','CP4',
                                'PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz','CPz','POz','Oz',])
 
    
            raw.pick_types(  meg = False, eeg = True, ecg = False)
        






        # interpolate channels based on thevariance of the channels 
        raw, badchans  = mark_bad_channels_interpolate(raw)
        # raw.plot_psd(tmax=250, average=False)

        raw._data = mne.filter.notch_filter(raw._data, raw.info['sfreq'], 50, notch_widths =2, phase='zero'  )
        # Creating epochs
        (events_from_annot, event_dict) = mne.events_from_annotations(raw)
        u, indices = np.unique(events_from_annot[:,0], return_index=True)
        events_from_annot_unique = events_from_annot[indices]
        event_unique, event_unique_ind  = np.unique(events_from_annot_unique[:,2], return_index=True)
        
        # Create epochs based on the events, from -1 to 1s
        # Set the baseline to None, because mne suggests to do a baseline correction after ICA
        epochs = mne.Epochs(raw, events_from_annot_unique, event_id=event_dict,
                            tmin=-1, tmax=1, reject=None, preload=True,  baseline=None, on_missing = 'ignore')
        
        

        
        
        # cubic interpolation for some subjects that have sth like a TMS artifact
        for sub_name, sub_name_v in enumerate(stim_artifact_subs):   
            print( sub_name, sub_name_v)
            if (f.parts[-1][9:13] == sub_name_v):
                for l_stim_art in range(len(stim_artifact_subs[f.parts[-1][9:13]])): 
                    if (f.parts[-1][9:13] == sub_name_v and stim_artifact_subs[f.parts[-1][9:13]][l_stim_art] ==  v )==True:
                        epochs_interpolated = cubic_interp(epochs, win=[-40, 10])
            else:
                epochs_interpolated = epochs


        #epochs_interpolated = epochs

        # Filtering
        # bandstop (48-52 Hz) filter data. lfreq higher than hfreq makes a bandstop filter
        epochs_interpolated.filter(1, 45, method='iir', verbose=0, iir_params=None, pad = 'reflect_limited') # If iir_params is none for iir filter then it will consider butterworth of order 4
    

        
        

        
        evokeds = epochs_interpolated.average()
        evokeds.plot(spatial_colors = True, gfp=True)

        ar = AutoReject(n_interpolate=[0])
        epochs_ar, reject_log_1 = ar.fit(epochs_interpolated).transform(epochs_interpolated, return_log = True)   

        #Applying ICA after filtering and before baseline correction 
        data_ica  = clean_dataset(epochs_ar)
        epochs_cleaned = data_ica['eeg']

        epochs_b = epochs_cleaned.apply_baseline(baseline=(-0.9, -0.1))
        evokeds = epochs_b.average()

        all_times = np.arange(0, 0.8, 0.02)
        fig_topo = evokeds.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
        fig_erp = evokeds.plot(spatial_colors = True, gfp=True)
        fig_erp.set_size_inches((20, 8))
        plt.plot(raw._data[4,:])
        plt.show()
        
        

        epochs_b.save(save_folder + str(f.parts[-1][4:14]) + v +'_epo.fif', overwrite = True, split_size='2GB')
        fig_topo.savefig(save_folder_figs +  str(f.parts[-1][4:14]) + v + '_topo' + '.svg')
        fig_erp.savefig(save_folder_figs +  str(f.parts[-1][4:14]) + v + '_erp'  + '.svg')

        
        
    
        if len(all_possible_stim_site)>1:
            if f.parts[-1] == 'TMS_NMES_RaPa_healthy_old2.xdf':
                pulses_8_phases = [pulses_8_phases[0], 'nan']
                
            cols = ['Sub_info','Pulse_8_phases', 'Pulse_0', 'Pulse_180', 'Pulse_switch']
            lists_pluses.append([f.parts[-1][9:13] + '_' + all_possible_stim_site[i], pulses_8_phases[0],  pulses_0[0] , pulses_180[0], switch_ind[0]])
            lists_pluses.append([f.parts[-1][9:13] + '_' + all_possible_stim_site[i], pulses_8_phases[1],  pulses_0[1] , pulses_180[1], switch_ind[-1]])

        if f.parts[-1] == 'TMS_NMES_RaPa_healthy_old1.xdf':
            cols = ['Sub_info','Pulse_8_phases', 'Pulse_0', 'Pulse_180', 'Pulse_switch']
            lists_pluses.append([f.parts[-1][9:13] + '_' + all_possible_stim_site[0], pulses_8_phases[0],  'nan' , 'nan', 'nan'])
 
    
    df_pulses = pd.DataFrame(lists_pluses, columns=cols )     
    df_pulses.to_csv (save_folder +'Subject_ pulse.csv', header=True) 
        
    plt.plot(raw._data[0,:])
    plt.show()


def clustering_channels_max(n_sub, win_erps, exdir_epoch, thresholds, labels, name, v_bar, save_folder):
    
    
    files = Path(exdir_epoch).glob('*epo.fif*')
    #plt.close('all')
    #idx =263
    
    unique_phases = np.arange(0, 360, 45)
    stim_int = np.arange(2, 18, 2)
    peaks = np.zeros([n_sub, 64, len(labels), len(unique_phases), len(stim_int)])     
    a = np.zeros([64, len(labels)])

    

    
    for ifiles, f in enumerate(files):
        print(ifiles, f)
        epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_amp_mod = epochs._data[:,:,1001:] - epochs._data[:,:,0:1000]
        # making mne epoch structure
        epochs = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, on_missing='ignore')

        #non_central_ch = flipping_ch(epochs.info['ch_names'])  
        epochs_bystimandphase = {} 
        erp_bystimandphase = {} 
        peaks_bystimandphase = {}
     
        
        for istim, stim in enumerate(stim_int):
            epochs_bystimandphase[str(stim)] = {}
            erp_bystimandphase[str(stim)] = {} 
            peaks_bystimandphase[str(stim)] = {} 
          
            for iphase, phase in enumerate(unique_phases):
                sel_idx = Select_Epochs_intensity_phase(epochs, stim, phase)
                epochs_bystimandphase[str(stim)][str(phase)] = epochs[sel_idx]
                erp_bystimandphase[str(stim)][str(phase)]  = epochs_bystimandphase[str(stim)][str(phase)].average() 
# =============================================================================
#                 # Flipping the hemisphere
#                 if name == 'V2_R':
#                    for i_ind in non_central_ch:
#                        erp_bystimandphase[str(stim)][str(phase)] ._data[i_ind:i_ind+2,:] = np.flipud(erp_bystimandphase[str(stim)][str(phase)] ._data[i_ind:i_ind+2,:])
#                    else:
# 
#                          erp_bystimandphase[str(stim)][str(phase)]=   erp_bystimandphase[str(stim)][str(phase)]
# =============================================================================
                        
                for ipeak, peak in enumerate(labels):
              
                    
                    if ipeak == 0:    #P45
                        peaks_bystimandphase[str(stim)][str(phase)] = np.max((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                    
                    elif  ipeak == 1: #N60
                        peaks_bystimandphase[str(stim)][str(phase)] = np.min((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[1,0]:win_erps[1,1]]),1)
                    
                    elif  ipeak == 2: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.max((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[2,0]:win_erps[2,1]]),1)


                    if str(erp_bystimandphase[str(stim)][str(phase)].comment) == str(''):    # To remove none arrays after selecting epochs
                        peaks_bystimandphase[str(stim)][str(phase)] = np.zeros(64) 
                       
                    else:
                        peaks[ifiles, :, ipeak, iphase, istim] = (peaks_bystimandphase[str(stim)][str(phase)] )
             
    

              
                

    adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs_bystimandphase[str(stim)][str(phase)].info , 'eeg')
    clusters, mask, pvals = permutation_cluster_max(peaks, adjacency_mat, thresholds)         
    nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)    
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters)):
        allclusters[:,p] = clusters[p][0]
    # set all other t values to 0 to focus on clusters
    allclusters[mask==False] = 0
    ch_names = epochs.ch_names
    # this is putting the 5-dim data structure in the right format for performing the sine fits
    
    for p in range(len(clusters)):
        a[:,p] = clusters[p][0]
        
    #combine labels 2 and 3, they are the same component. just different cluster    
    #a_com = a[:,[0, 1, 2, 4]]
    a[a > v_bar] = v_bar
    a[a < -v_bar] = -v_bar
    fig = plot_topomap_peaks_second_v(name, a, mask, ch_names, pvals,[-v_bar-3,v_bar+3], epochs.info, i_intensity = 'all')
    #fig = plot_topomap_peaks_second_v(a, mask, ch_names, pvals,[-v_bar,v_bar], epochs.info, i_intensity = 'all')
    
    
    # Name and indices of the EEG electrodes that are in the biggest cluster
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask[:,p] == 1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask[:,p] == 1)[0]])
        

    fig.savefig(save_folder + name + '_max.svg')    

    return all_ch_names_biggets_cluster, pvals, a, mask, epochs.info, np.mean(peaks, (-2, -1)), peaks







def permutation_cluster_max(peaks, adjacency_mat, thresholds):

    
    # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2, -1))
    

    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    mask = np.zeros([nchans, npeaks])
    pvals = np.zeros([npeaks])
    clusters = []
   

    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

    for p in range(npeaks):
        cluster = mne.stats.permutation_cluster_1samp_test( (mean_peaks[:,:,p]), out_type='mask',
                                                           adjacency=adjacency_mat, threshold=thresholds[p],
                                                           n_permutations=1000)
        t_sum = np.zeros([len(cluster[1])])
        # get the sum of the tvalues for each of the 
        # clusters to choose the main cluster 
        # (take magnitude to treat negative and positive cluster equally)
        for c in range(len(cluster[1])):
            t_sum[c] = np.abs(sum(cluster[0][cluster[1][c]]))
    
       
        
        
        if len(t_sum) > 0:
                mask[:,p] = cluster[1][np.argmax(t_sum)]
                pvals[p] = min(cluster[2])
            

        clusters.append(cluster)         
        

    return clusters, mask, pvals








def clustering_channels_each_intensity_max(i_intensity, n_sub, win_erps, exdir_epoch, thresholds, labels, name, save_folder):
    
    
    files = Path(exdir_epoch).glob('*epo.fif*')
    #plt.close('all')
    #idx =263
    
    unique_phases = np.arange(0, 360, 45)
    stim_int = np.arange(2, 18, 2)
    peaks = np.zeros([n_sub, 64, len(labels), len(unique_phases), len(stim_int)])     
    a = np.zeros([64, len(labels)])

    

    
    for ifiles, f in enumerate(files):
        print(ifiles, f)
        epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_amp_mod = epochs._data[:,:,1001:] - epochs._data[:,:,0:1000]
        # making mne epoch structure
        epochs = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, on_missing='ignore')

        #non_central_ch = flipping_ch(epochs.info['ch_names'])  
        epochs_bystimandphase = {} 
        erp_bystimandphase = {} 
        peaks_bystimandphase = {}
     
        
        for istim, stim in enumerate(stim_int):
            epochs_bystimandphase[str(stim)] = {}
            erp_bystimandphase[str(stim)] = {} 
            peaks_bystimandphase[str(stim)] = {} 
          
            for iphase, phase in enumerate(unique_phases):
                sel_idx = Select_Epochs_intensity_phase(epochs, stim, phase)
                epochs_bystimandphase[str(stim)][str(phase)] = epochs[sel_idx]
                erp_bystimandphase[str(stim)][str(phase)]  = epochs_bystimandphase[str(stim)][str(phase)].average() 
# =============================================================================
#                 # Flipping the hemisphere
#                 if name == 'V2_R':
#                    for i_ind in non_central_ch:
#                        erp_bystimandphase[str(stim)][str(phase)] ._data[i_ind:i_ind+2,:] = np.flipud(erp_bystimandphase[str(stim)][str(phase)] ._data[i_ind:i_ind+2,:])
#                    else:
# 
#                          erp_bystimandphase[str(stim)][str(phase)]=   erp_bystimandphase[str(stim)][str(phase)]
# =============================================================================
                        
                for ipeak, peak in enumerate(labels):
              
                    
                    if ipeak == 0:    #P45
                        peaks_bystimandphase[str(stim)][str(phase)] = np.max((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                    
                    elif  ipeak == 1: #N60
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[1,0]:win_erps[1,1]]),1)
                    
                    elif  ipeak == 2: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[2,0]:win_erps[2,1]]),1)

                    elif  ipeak == 3: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[3,0]:win_erps[3,1]]),1)
                   

                    if str(erp_bystimandphase[str(stim)][str(phase)].comment) == str(''):    # To remove none arrays after selecting epochs
                        peaks_bystimandphase[str(stim)][str(phase)] = np.zeros(64) 
                       
                    else:
                        peaks[ifiles, :, ipeak, iphase, istim] = (peaks_bystimandphase[str(stim)][str(phase)] )
             
    

              
                

    adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs_bystimandphase[str(stim)][str(phase)].info , 'eeg')
    clusters, mask, pvals = permutation_cluster_each_intensity_max(peaks, adjacency_mat, thresholds, i_intensity)         
    nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)    
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters)):
        allclusters[:,p] = clusters[p][0]
    # set all other t values to 0 to focus on clusters
    allclusters[mask==False] = 0
    ch_names = epochs.ch_names
    # this is putting the 5-dim data structure in the right format for performing the sine fits
    
    for p in range(len(clusters)):
        a[:,p] = clusters[p][0]
        
    #combine labels 2 and 3, they are the same component. just different cluster    
    #a_com = a[:,[0, 1, 2, 4]]
    a[a > 5] = 3
    a[a < -5] = -3
    fig = plot_topomap_peaks_second_v(plot_topomap_peaks_second_v, a, mask, ch_names, pvals,[-5,5], epochs.info , i_intensity)
    
    
    
    # Name and indices of the EEG electrodes that are in the biggest cluster
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask[:,p] == 1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask[:,p] == 1)[0]])
        

    fig.savefig( save_folder + name +'_' +f'{ (i_intensity+1)*2}'+ '_max.svg')    

    return all_ch_names_biggets_cluster, pvals, a, mask, epochs.info








def permutation_cluster_each_intensity_max(peaks, adjacency_mat, thresholds, i_intensity):
   
    
    # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2))
    

    # get matrix dimensions
    nsubj, nchans, npeaks, _ = np.shape(mean_peaks)
    mask = np.zeros([nchans, npeaks])
    pvals = np.zeros([npeaks])
    clusters = []
   

    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

    for p in range(npeaks):
        cluster = mne.stats.permutation_cluster_1samp_test( (mean_peaks[:,:,p, i_intensity]), out_type='mask',
                                                           adjacency=adjacency_mat, threshold=thresholds[p],
                                                           n_permutations=1000)
        t_sum = np.zeros([len(cluster[1])])
        # get the sum of the tvalues for each of the 
        # clusters to choose the main cluster 
        # (take magnitude to treat negative and positive cluster equally)
        for c in range(len(cluster[1])):
            t_sum[c] = np.abs(sum(cluster[0][cluster[1][c]]))
    
        # store the maximal cluster size for each iteration 
        # to later calculate p value
        # if no cluster was found, put in 0
        
        
        
        if len(t_sum) > 0:
                mask[:,p] = cluster[1][np.argmax(t_sum)]
                pvals[p] = min(cluster[2])
            

        clusters.append(cluster)         
        

    return clusters, mask, pvals
















def XDF_correct_time_stamp_reject_pulses(f):
    
    
    marker = pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    brainvision = pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    edcix = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == 'EDC_R'][0]
    edcdat = brainvision['time_series'][:,edcix]
    out = {'pulse_BV':[], 'drop_idx_list': []}
    
    # pulses creates a list of the indices of the marker timestamps for the stimulation condition trials only
    # i.e., excluding the vigilance task trials
    pulses = [i for i,m in enumerate(marker['time_series']) if "\"stim_type\": \"TMS\"" in m[0]]

    # pulseinfo contains a list of the stim.condition time stamps and descriptions
    # each item in the list contains a list with the size 2: pulseinfo[i][0] is the timestamp corresponding with the index i from pulses,
    # pulseinfo[i][1] contains the corresponding stimulus description (i.e., stim phase and freq, etc.)
    pulseinfo = [[np.searchsorted(brainvision['time_stamps'], marker['time_stamps'][p]), marker['time_series'][p]] for p in pulses]
    n=0
    
    for i,p in enumerate(pulseinfo):
        pulse_idx = pulses[pulseinfo.index(p)]
        sample = p[0]

        # For the NMES study, we use the ECD_R data to identify the artifact
        # and we use a time window around the onset of the original reizmarker_timestamp: [sample-1500:sample+1500]
        onset = sample-1500
        offset = sample+1500
        edcep = edcdat[onset:offset]
        dmy= np.abs(stats.zscore(edcep))
        tartifact = np.argmax(dmy)
        
        # edcep contains 3000 timepoints or samples (-1500 to +1500 samples around the original rm_marker)
        # so, if tartifact is < 1500, the new marker is in the period before the original marker
        # if tartifact is >1500, the new marker is in the period after the original marker      
        corrected_timestamp = sample - 1500 + tartifact
 
        #print('the original marker ts was: ' + str(sample)+' and the corrected ts is: '+str(corrected_timestamp))
        
        # the section below is to check for trials where no clear stimulation artifact is present
        # a list of indices is created and saved in out['drop_idx_list'], to be used to reject 
        # these epochs when the preprocessing in MNE is started
        if max(dmy) < 3:
            n+=1
            out['drop_idx_list'].append(pulse_idx)
        out['pulse_BV'].append(corrected_timestamp)
    _, _, pulses_ind_drop = np.intersect1d(out['drop_idx_list'], pulses, return_indices=True)

# =============================================================================
#     pulses_ind_drop_filename = 'pulses_ind_drop_'+ str(f.parts[-3])+'_'+str(f.parts[-1][-8:-4])+'.p'
#     with open(str(save_folder_pickle) +pulses_ind_drop_filename, 'wb') as fp:
#         pickle.dump(pulses_ind_drop, fp, protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================
        
        
    """        
    Next, replace the original timestamps in the marker stream with the new ones.   
    - the original markers are stored in marker['time_stamps']
    - the new time stamp values are based on the brainvision['time_stamps'] values that 
    correspond with the brainvision['time_stamps'] index as stored in out['pulse_BV']
        E.g., 
        corrected_timestamp = 50961
        In [9]: brainvision['time_stamps'][50961]
        Out[9]: 374680.57453827135
        
    IMPORTANT:the values in corrected_timestamp (and pulse info) refer to the index of the timestamp, not
    the actual time value, of a timestamp in brainvision
    """
    
    marker_corrected = marker
    
    for i in range(len(pulses)):
        # for the stim.condition time stamps (corresponding to the indices stored in pulses)
        # replace original reizmarker (rm) timestamp value with the corrected timestamp value based on the EDC artifact (corrected_timestamp)
        rm_timestamp_idx = pulses[i]
        brainvision_idx = out['pulse_BV'][i]
        rm_timestamp_new_value = brainvision['time_stamps'][brainvision_idx] 
                
        #print('old value: '+str(marker['time_stamps'][pulses[i]]))
        # replace original stimulus onset time stamp with the new timestamp value
        marker_corrected['time_stamps'][rm_timestamp_idx] = rm_timestamp_new_value
        #print('new value: '+str(marker['time_stamps'][pulses[i]]))

        

    #### convert brainvision and corrected marker stream into a fif file that can be read by MNE ###    

    #marker_corrected = marker    #pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    data = brainvision   #pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    marker_corrected['time_stamps'] -= data['time_stamps'][0] #remove clock offset
    
    channel_names = [c['label'][0] for c in data['info']['desc'][0]['channels'][0]['channel'] ]
    sfreq = int(data['info']['nominal_srate'][0])
    types = ['eeg']*64
    types.extend(['emg']*(len(channel_names)-64)) #64 EEG chans, rest is EMG/EKG
    info = mne.create_info(ch_names = channel_names, sfreq = sfreq, ch_types = types)
    raw = mne.io.RawArray(data = data['time_series'].T, info = info)
    
    if len(marker_corrected['time_stamps']) > 1:
        descs = [msg[0] for msg in marker_corrected['time_series']]
        ts = marker_corrected['time_stamps']
        
        sel = [i for i,v in enumerate(descs) if "TMS" in v]
        descs = [descs[i] for i in sel]
        
        ts = [ts[i] for i in sel]
        
        shortdescs = [json.loads(msg)['freq'] + 'Hz_' + json.loads(msg)['phase'] for msg in descs]

        anno = mne.Annotations(onset = ts, duration = 0, description = shortdescs)
        raw = raw.set_annotations(anno)
        
    ts_new = np.delete(ts, pulses_ind_drop)
    shortdescs_new = np.delete(shortdescs, pulses_ind_drop)
    anno = mne.Annotations(onset = ts_new, duration = 0, description = shortdescs_new)
    raw = raw.set_annotations(anno)      
    #print(len(ts), len(ts_new))
    #print(str(f.parts[-3]))
   
          


    return raw, len(ts) , len(ts_new)                  




from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS

def regress_out_pupils(raw, ocular_channels = ['Fpz', 'Fp1', 'Fp2', 'AF7', 'AF8'], method = 'PCA'):
    
    """
    raw: Continuous raw data in MNE format
    ocular_channels: can be labels of EOG channels or EEG channels close to the
        eyes if no EOG was recorded
    method: how to combine the ocular channels. Can be 'PCA', 'mean', or 'median'.
    """
    
    raw_data = raw.get_data(picks = 'eeg')
    ocular_data = raw.get_data(picks = ocular_channels)
    
    if method == 'PCA':
        pca = PCA()
        comps = pca.fit_transform(ocular_data.T)
        ocular_chan = comps[:,0]
    elif method == 'mean':
        ocular_chan = np.mean(ocular_data, axis = 0)
    elif method == 'median':
        ocular_chan = np.median(ocular_data, axis = 0)
    
    for ch in range(raw_data.shape[0]):
        m = OLS(raw_data[ch,:], ocular_chan)
        raw_data[ch,:] -= m.fit().predict()
    raw._data[:raw_data.shape[0],:] = raw_data
    return raw





def mark_bad_channels_interpolate(raw):

    """
    Detects channels above a certain threshold when looking at zscored data.
    Plots time series with pre-detected channels marked in red.
    Enables user to mark bad channels interactively and saves selection in raw object.

    Args:
        raw : MNE raw object with EEG data
        ch_names (list): list of strings with channel names
        threshold (float, int): threshold based on which to detect outlier channels 
                                (maximal zscored absolute standard deviation). Defaults to 1.5.

    Returns:
        MNE raw object: raw, with bad channel selection (bads) updated
    """


    ch_names = raw.info['ch_names']


    # plotting of channel variance
    vars = np.var(raw._data.T, axis=0)
    badchans_threshold = np.where(np.abs(zscore(vars)) > 1.5)
    #badchans = visual_inspection(vars)
    raw.info['bads'] = [ch_names[i] for i in list(badchans_threshold[0])]

    # filtering and plotting of raw data with marked bad channels
    # bandpass and bandstop filter data
# =============================================================================
#     raw.filter(0.5, 49, method='iir', verbose=0)
#     raw._data = mne.filter.notch_filter(raw._data, raw.info['sfreq'], 50, notch_widths=2,
#                                         picks=[i for i, ch in enumerate(ch_names) if ch not in raw.info['bads']], 
#                                         phase='zero', verbose=0)
# =============================================================================

    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    ch_names = raw.info['ch_names']
    badchans_threshold  = raw.info['bads']
    raw_eeg_interp = raw.interpolate_bads(reset_bads=True)
    
    

    return raw_eeg_interp, badchans_threshold




def MNE_raw_format(eegdata, ch_names, sfreq):

    """
    Helps get EEG data into MNE raw object format by using some default values.
    Channel types all defined as EEG and default montage used.

    Args:
        eegdata (numpy array): EEG data array, timepoints*channels
        ch_names (list): list of strings with channel names
        sfreq (float, int): sampling frequency

    Returns:
        raw : instance of MNE raw
    """
    
    import mne
    
    ch_types = ['eeg']*len(ch_names)
    
    info = mne.create_info(ch_names=ch_names, 
                           sfreq=sfreq, 
                           ch_types=ch_types,
                           verbose=0)

    raw = mne.io.RawArray(np.transpose(eegdata), info, verbose=0)

    raw.set_montage(mne.channels.make_standard_montage('standard_1005'), verbose=0)

    return raw




def visual_inspection(x, indexmode = 'exclude'):
    """
    Allows you to visually inspect and exclude elements from an array.
    The array x typically contains summary statistics, e.g., the signal
    variance for each trial.
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    import numpy as np

    x = np.array(x)
    x = x.flatten()
    nanix = np.zeros(len(x))
    
    
    
    
    def line_select_callback(eclick, erelease):
        """
        Callback for line selection.
    
        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    

    fig, current_ax = plt.subplots()                 # make a new plotting range
    # plt.plot(np.arange(len(x)), x, lw=1, c='b', alpha=.7)  # plot something
    current_ax.plot(x, 'b.', alpha=.7)  # plot something

    print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    RS = RectangleSelector(current_ax, line_select_callback,
                                    drawtype='box', useblit=True,
                                    button=[1],  # don't use middle button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
    RSinv = RectangleSelector(current_ax, line_select_callback,
                                drawtype='box', useblit=True,
                                button=[3],  # don't use middle button
                                minspanx=5, minspany=5,
                                spancoords='pixels',
                                interactive=True)
    plt.connect('key_press_event', (RS, RSinv))

    while plt.fignum_exists(1):
        plt.cla()
        current_ax.set_ylim([np.min(x[np.where(nanix == 0)[0]]), 1.1*np.max(x[np.where(nanix == 0)[0]])])
        current_ax.plot(x, 'b.', alpha=.7)  # plot something
        if np.sum(nanix) > 0:
            current_ax.plot(np.squeeze(np.where(nanix == 1)), x[np.where(nanix == 1)], 'w.', alpha=.7)  # plot something

        fig.show()
        plt.pause(.1)
        if plt.fignum_exists(1):
            plt.waitforbuttonpress(timeout = 2)
            
            if (RS.geometry[1][1] > 1):
                exclix = np.where((x > min(RS.geometry[0])) & (x < max(RS.geometry[0])))[0]
                exclix = exclix[np.where((exclix > min(RS.geometry[1])) & (exclix < max(RS.geometry[1])))]
                nanix[exclix] = 1
            if (RSinv.geometry[1][1] > 1):
                exclix = np.where((x > min(RSinv.geometry[0])) & (x < max(RSinv.geometry[0])))[0]
                exclix = exclix[np.where((exclix > min(RSinv.geometry[1])) & (exclix < max(RSinv.geometry[1])))]
                nanix[exclix] = 0
            if not plt.fignum_exists(1):
                break
            else:
                plt.pause(.1)
        else:
            plt.pause(.1)
            break
    if indexmode == 'exclude':
    	return np.where(nanix == 1)[0]
    elif indexmode == 'keep':
    	return np.where(nanix == 0)[0]
    else:
    	raise ValueError
        
        
#%% ICA

import mne
from pathlib import Path


from autoreject import AutoReject, get_rejection_threshold


# from .eeg_utils import *


# https://github.com/HemuManju/human-effort-classification-eeg/blob/223e320e7201f6c93cbe8f9728e401d7199453a2/src/data/clean_eeg_dataset.py
def autoreject_repair_epochs(epochs, reject_plot=False):
    """Rejects the bad epochs with AutoReject algorithm
    Parameter
    ----------
    epochs : Epoched, filtered eeg data
    Returns
    ----------
    epochs : Epoched data after rejection of bad epochs
    """
    # Cleaning with autoreject
    #picks = mne.pick_types(epochs.info, eeg=True)  # Pick EEG channels
    ar = AutoReject(n_interpolate = np.array([1, 4, 32]),
    consensus= np.linspace(0, 1.0, 11),
                    verbose=False)

    #ar = AutoReject(n_interpolate=[1], consensus_percs=[0.6])

    cleaned_epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    if reject_plot:
        reject_log.plot_epochs(epochs, scalings=dict(eeg=40e-6))

    return cleaned_epochs


def append_eog_index(epochs, ica):
    """Detects the eye blink aritifact indices and adds that information to ICA
    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    ica    : ica object from mne
    Returns
    ----------
    ICA : ICA object with eog indices appended
    """
    # Find bad EOG artifact (eye blinks) by correlating with Fp1
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp1',
                                             verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.65]
    ica.exclude += id_eog

    # Find bad EOG artifact (eye blinks) by correlation with Fp2
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp2',
                                             verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.75]
    ica.exclude += id_eog

    return ica


def clean_with_ica(epochs, show_ica=False):
    """Clean epochs with ICA.
    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    Returns
    ----------
    ica     : ICA object from mne
    epochs  : ICA cleaned epochs
    """

    picks = mne.pick_types(epochs.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=False,
                           exclude='bads')
    ica = mne.preprocessing.ICA(n_components=None,
                                method="fastica",
                                verbose=False)
    # Get the rejection threshold using autoreject
    #reject_threshold = get_rejection_threshold(epochs)
    ica.fit(epochs, picks=picks)
    #ica.fit(epochs[~reject_log.bad_epochs])
    
    ica = append_eog_index(epochs, ica)  # Append the eog index to ICA
    if show_ica:
        ica.plot_components(inst=epochs)
    clean_epochs = ica.apply(epochs.copy())  # Remove selected components from the signal.

    return clean_epochs, ica


# =============================================================================
# def clean_dataset(epochs):
#     """Create cleaned dataset (by running autoreject and ICA)
#     with each subject data in a dictionary.
#     Parameter
#     ----------
#     subject : string of subject ID e.g. 7707
#     trial   : HighFine, HighGross, LowFine, LowGross
#     Returns
#     ----------
#     clean_eeg_dataset : dataset of all the subjects with different conditions
#     """
#     data  = {}
#     ica_epochs, ica = clean_with_ica(epochs)
#     repaired_eeg = autoreject_repair_epochs(ica_epochs)
#     data['eeg'] = ica_epochs
#     data['ica'] = ica
# 
# 
#     return data
#         
# =============================================================================

def clean_dataset(epochs):
    """Create cleaned dataset (by running autoreject and ICA)
    with each subject data in a dictionary.
    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross
    Returns
    ----------
    clean_eeg_dataset : dataset of all the subjects with different conditions
    """
    data  = {}
    ica_epochs, ica = clean_with_ica(epochs)
    repaired_eeg = autoreject_repair_epochs(ica_epochs)
    data['eeg'] = repaired_eeg
    data['ica'] = ica


    return data
        






def cubic_interp(epochs, win):

    from scipy import interpolate

    # convert interpolation window to indices in epochs
    idx1 = np.argmin(np.abs(epochs.times - win[0]*0.001))
    idx2 = np.argmin(np.abs(epochs.times - win[1]*0.001))

    # get timepoints in seconds and delete those that should be interpolated
    x = epochs.times
    x = np.delete(x, np.s_[idx1:idx2], 0)
    

    for i, epoch in enumerate(epochs):

        y = np.delete(epoch, np.s_[idx1:idx2], -1)
        p = interpolate.interp1d(x, y, kind='cubic')

        # get the interpolation values for the timepoints of interest
        interp_values = p.__call__(epochs.times[idx1:idx2])

        # for each epoch replace timepoints in epochs object
        epochs._data[i, :, idx1:idx2] = interp_values

    return epochs








def Select_Epochs_phase(epochs, phase):
    """ 
    this function selects epochs based on the phases and ignores intensities 
    out puts: index of the epochs with the desired phased

    """
    
    index_list = []
    events_array = epochs.events
    event_id_dict = epochs.event_id
    # example o event description for acute NMES study: “freq”: “4”, “phase”: “0”
    phase_to_select = str(phase) 
    
    
    for i in range(len(events_array)):
        event_code = events_array[i,2]
        event_id_key = list(event_id_dict.keys())[list(event_id_dict.values()).index(event_code)]
        
        if event_id_key.find("I") == -1: # To exclude IO events (extra 0 and 180 pulses)
                #if (freq_to_select in str(event_id_key[:(event_id_key.find('_') -2)])) == True and (phase_to_select in str(event_id_key[event_id_key.find('_') + 1:])) == True:
                if   (phase_to_select == str(int(float(event_id_key[event_id_key.find('m') + 4:])))) :    
                    index_list.append(i)      
                else:
                    continue

    return index_list
    


def Select_Epochs_intensity_phase(epochs, stim, phase):
    """ 
    this is a function that will identify epochs based on their key (a string) in event_id, 
    which describes the stimulation condition
        
    selection depends on the frequency and the phase of interest
        
    the function returns a list of event indices, that only includes the indices of epochs that contained 
    stimulation at the desired frequency and phase
        
        
    data: epochs data in MNE format
    freq: an integer number, this can be any number between 0 and 40 and depends on the frequencies
    that were stimulated in your study (and thus described in your event description (a string) in event_id)
    phase: an integer number, this can be any number between 0 and 360 and depends on the phases
    that were stimulated in your study (and thus described in your event description in event_id)
    """
    
    index_list = []
    events_array = epochs.events
    event_id_dict = epochs.event_id
    # example o event description for acute NMES study: “freq”: “4”, “phase”: “0”
    stim_to_select = str(stim) 
    phase_to_select = str(phase) 
    
    
    for i in range(len(events_array)):
        event_code = events_array[i,2]
        event_id_key = list(event_id_dict.keys())[list(event_id_dict.values()).index(event_code)]
        
        if event_id_key.find("I") == -1: # To exclude IO events (extra 0 and 180 pulses)
            if phase >= 0 and phase <=360:
                #if (freq_to_select in str(event_id_key[:(event_id_key.find('_') -2)])) == True and (phase_to_select in str(event_id_key[event_id_key.find('_') + 1:])) == True:
                if (stim_to_select == str(event_id_key[event_id_key.find('m') - (event_id_key.find('m')-event_id_key.find("'")-1):event_id_key.find('m')]))  and (phase_to_select == str(int(float(event_id_key[event_id_key.find('m') + 4:])))) :    
                    index_list.append(i)      
                else:
                    continue

    return index_list
    





def Select_Epochs_intensity(epochs, stim):
    """ 
    this is a function that will identify epochs based on their key (a string) in event_id, 
    which describes the stimulation condition
        
    selection depends on the frequency and the phase of interest
        
    the function returns a list of event indices, that only includes the indices of epochs that contained 
    stimulation at the desired frequency and phase
        
        
    data: epochs data in MNE format
    freq: an integer number, this can be any number between 0 and 40 and depends on the frequencies
    that were stimulated in your study (and thus described in your event description (a string) in event_id)
    phase: an integer number, this can be any number between 0 and 360 and depends on the phases
    that were stimulated in your study (and thus described in your event description in event_id)
    """
    
    index_list = []
    events_array = epochs.events
    event_id_dict = epochs.event_id
    # example o event description for acute NMES study: “freq”: “4”, “phase”: “0”
    stim_to_select = str(stim) 
    
    
    
    for i in range(len(events_array)):
        event_code = events_array[i,2]
        event_id_key = list(event_id_dict.keys())[list(event_id_dict.values()).index(event_code)]
        
        if event_id_key.find("I") == -1: # To exclude IO events (extra 0 and 180 pulses)
                #if (freq_to_select in str(event_id_key[:(event_id_key.find('_') -2)])) == True and (phase_to_select in str(event_id_key[event_id_key.find('_') + 1:])) == True:
                if (stim_to_select == str(event_id_key[event_id_key.find('m') - (event_id_key.find('m')-event_id_key.find("'")-1):event_id_key.find('m')])):    
                    index_list.append(i)      
                else:
                    continue

    return index_list








def flipping_ch(all_channels):

    '''     # Original channel indices: 0.Fp1   1.Fp2    2.F3   3.F4   4.C3   5.C4   6.P3   7.P4   8.O1    9.O2   10.F7
            #                           1.Fp1   0.Fp2    3.F3   2.F4   5.C3   4.C4   7.P3   6.P4   9.O1    8.O2   11.F7  
            #                     
            # An example of how it works for a simple Matrix
            A = np.arange(32).reshape((8,4))
            for i_ind in range(0, 5,2):
                A[i_ind:i_ind+2,:] = np.flipud(A[i_ind:i_ind+2,:])
                
                
                
            one point is the central channels need to stay the same, therefore we need to exclude them from the list of channels that are flipped    
                '''
    
    # Flipping the hemisphere
    
    central_channels = ['Fz','Cz', 'Pz', 'Iz', 'Fpz', 'CPz', 'POz', 'Oz']   
    central_channels_ind_even = []
    for ind_ch in central_channels:   
        central_channels_ind = [index for (index, item) in enumerate(all_channels) if item == ind_ch][0] 
        if (central_channels_ind % 2) == 0:
            central_channels_ind_even.append(central_channels_ind) # we only need even numbers       
            
    non_central_ch = np.setdiff1d(np.arange(0, 64, 2),central_channels_ind_even)
    return non_central_ch





def clustering_channels(n_sub, win_erps, exdir_epoch, thresholds, labels, name, save_folder):
    
   
    files = Path(exdir_epoch).glob('*epo.fif*')

    #files =  Path(exdir_epoch).glob('*BuUl_pre3_old2R_manually_epo.fif*')
    
    
    #plt.close('all')
    #idx =263
    
    unique_phases = np.arange(0, 360, 45)
    stim_int = np.arange(2, 18, 2)
    peaks = np.zeros([n_sub, 64, len(labels), len(unique_phases), len(stim_int)])     
    a = np.zeros([64, len(labels)])

    

    
    for ifiles, f in enumerate(files):
        print(ifiles, f)

        epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        epochs_p1 =  mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_amp_mod = epochs._data[:,:,1001:] - epochs._data[:,:,0:1000]
        # making mne epoch structure
        epochs = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, on_missing='ignore')

        #non_central_ch = flipping_ch(epochs.info['ch_names'])  
        epochs_bystimandphase = {} 
        epochs_bystimandphase_p1 = {}
        erp_bystimandphase = {} 
        peaks_bystimandphase = {}
        erp_bystimandphase_p1 = {}
     
        
        for istim, stim in enumerate(stim_int):
            epochs_bystimandphase_p1[str(stim)] = {}
            epochs_bystimandphase[str(stim)] ={}
            erp_bystimandphase[str(stim)] = {} 
            erp_bystimandphase_p1[str(stim)] = {}
            peaks_bystimandphase[str(stim)] = {} 
          
            for iphase, phase in enumerate(unique_phases):
                sel_idx = Select_Epochs_intensity_phase(epochs, stim, phase)
                epochs_bystimandphase_p1[str(stim)][str(phase)] = epochs_p1[Select_Epochs_intensity_phase(epochs_p1, stim, phase)] # This is because of stim artifact around 0
                erp_bystimandphase_p1[str(stim)][str(phase)]  = epochs_bystimandphase_p1[str(stim)][str(phase)].average() 
                epochs_bystimandphase[str(stim)][str(phase)] = epochs[sel_idx]
                erp_bystimandphase[str(stim)][str(phase)]  = epochs_bystimandphase[str(stim)][str(phase)].average() 
# =============================================================================
#                 # Flipping the hemisphere
#                 if name == 'V2_R':
#                    for i_ind in non_central_ch:
#                        erp_bystimandphase[str(stim)][str(phase)] ._data[i_ind:i_ind+2,:] = np.flipud(erp_bystimandphase[str(stim)][str(phase)] ._data[i_ind:i_ind+2,:])
#                    else:
# 
#                          erp_bystimandphase[str(stim)][str(phase)]=   erp_bystimandphase[str(stim)][str(phase)]
# =============================================================================
                        
                for ipeak, peak in enumerate(labels):
              
                    
                    if ipeak == 0:    #P45
                        if (f.parts[-2] == 'v2' and f.parts[-3] == 'Left'):
                            peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase_p1[str(stim)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                        else:
                            peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                    
                    elif  ipeak == 1: #N60
                            peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[1,0]:win_erps[1,1]]),1)
      
                    elif  ipeak == 2: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[2,0]:win_erps[2,1]]),1)

                    elif  ipeak == 3: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[3,0]:win_erps[3,1]]),1)


                    if str(erp_bystimandphase[str(stim)][str(phase)].comment) == str(''):    # To remove none arrays after selecting epochs
                        peaks_bystimandphase[str(stim)][str(phase)] = np.zeros(64) 
                       
                    else:
                        peaks[ifiles, :, ipeak, iphase, istim] = (peaks_bystimandphase[str(stim)][str(phase)] )
             
    

              
                

    adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs.info , 'eeg')
    clusters, mask, pvals = permutation_cluster(peaks, adjacency_mat, thresholds)         
    nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)    
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters)):
        allclusters[:,p] = clusters[p][0]
    # set all other t values to 0 to focus on clusters
    allclusters[mask==False] = 0
    ch_names = epochs.ch_names
    # this is putting the 5-dim data structure in the right format for performing the sine fits
    
    for p in range(len(clusters)):
        a[:,p] = clusters[p][0]
        
    #combine labels 2 and 3, they are the same component. just different cluster    
    #a_com = a[:,[0, 1, 2, 4]]
    a[a > 4] = 4
    a[a < -4] = -4
    if name == 'R_v6':
        a[:,2][a[:,2] > 3] = 2
    fig = plot_topomap_peaks_second_v(name, a, mask, ch_names, pvals,[-5,5], epochs.info, i_intensity = 'all')
   
    
    
    # Name and indices of the EEG electrodes that are in the biggest cluster
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask[:,p] == 1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask[:,p] == 1)[0]])
        

    fig.savefig(save_folder + name + '.svg')    

    return all_ch_names_biggets_cluster, pvals, a, mask, epochs.info, np.mean(peaks, (-2, -1))



def channel_names():
    ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','Iz','FC1','FC2','CP1',
    'CP2','FC5','FC6','CP5','CP6','FT9','FT10','TP9','TP10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4','CP3','CP4',
    'PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz','CPz','POz','Oz']
    return ch_names



def permutation_cluster(peaks, adjacency_mat, thresholds):
    #from sklearn import preprocessing
    
    # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2, -1))
    

    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    mask = np.zeros([nchans, npeaks])
    pvals = np.zeros([npeaks])
    clusters = []
   

    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

    for p in range(npeaks):
        #if p == 0:
        mean_peaks[:, [63, 9, 59, 15, 29, 8, 58, 19, 14], ] = 0.006 # control group


        cluster = mne.stats.permutation_cluster_1samp_test(zscore((mean_peaks[:,:,p]), axis =1), out_type='mask',
                                                           adjacency=adjacency_mat, threshold=thresholds[p],
                                                           n_permutations=1000)
        

            
        t_sum = np.zeros([len(cluster[1])])
        # get the sum of the tvalues for each of the 
        # clusters to choose the main cluster 
        # (take magnitude to treat negative and positive cluster equally)
        for c in range(len(cluster[1])):
            t_sum[c] = np.abs(sum(cluster[0][cluster[1][c]]))
    
       
        
        
        if len(t_sum) > 0:
                mask[:,p] = cluster[1][np.argmax(t_sum)]
                pvals[p] = min(cluster[2])
            

        clusters.append(cluster)         
        

    return clusters, mask, pvals




def permutation_cluster_each_intensity(peaks, adjacency_mat, thresholds, i_intensity):
   
    
    # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2))
    

    # get matrix dimensions
    nsubj, nchans, npeaks, _ = np.shape(mean_peaks)
    mask = np.zeros([nchans, npeaks])
    pvals = np.zeros([npeaks])
    clusters = []
   

    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

    for p in range(npeaks):
        cluster = mne.stats.permutation_cluster_1samp_test( zscore(mean_peaks[:,:,p, i_intensity], axis =1), out_type='mask',
                                                           adjacency=adjacency_mat, threshold=thresholds[p],
                                                           n_permutations=1000)
        t_sum = np.zeros([len(cluster[1])])
        # get the sum of the tvalues for each of the 
        # clusters to choose the main cluster 
        # (take magnitude to treat negative and positive cluster equally)
        for c in range(len(cluster[1])):
            t_sum[c] = np.abs(sum(cluster[0][cluster[1][c]]))
    
        # store the maximal cluster size for each iteration 
        # to later calculate p value
        # if no cluster was found, put in 0
        
        
        
        if len(t_sum) > 0:
                mask[:,p] = cluster[1][np.argmax(t_sum)]
                pvals[p] = min(cluster[2])
            

        clusters.append(cluster)         
        

    return clusters, mask, pvals




def clustering_channels_each_intensity(i_intensity, n_sub, win_erps, exdir_epoch, thresholds, labels, name, save_folder):
    
    
    files = Path(exdir_epoch).glob('*epo.fif*')
    #plt.close('all')
    #idx =263
    
    unique_phases = np.arange(0, 360, 45)
    stim_int = np.arange(2, 18, 2)
    peaks = np.zeros([n_sub, 64, len(labels), len(unique_phases), len(stim_int)])     
    a = np.zeros([64, len(labels)])

    

    
    for ifiles, f in enumerate(files):
        print(ifiles, f)
        epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        #epochs_amp_mod = epochs._data[:,:,1001:] - epochs._data[:,:,0:1000]
        # making mne epoch structure
        #epochs = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, on_missing='ignore')

        #non_central_ch = flipping_ch(epochs.info['ch_names'])  
        epochs_bystimandphase = {} 
        erp_bystimandphase = {} 
        peaks_bystimandphase = {}
     
        
        for istim, stim in enumerate(stim_int):
            epochs_bystimandphase[str(stim)] = {}
            erp_bystimandphase[str(stim)] = {} 
            peaks_bystimandphase[str(stim)] = {} 
          
            for iphase, phase in enumerate(unique_phases):
                sel_idx = Select_Epochs_intensity_phase(epochs, stim, phase)
                epochs_bystimandphase[str(stim)][str(phase)] = epochs[sel_idx]
                erp_bystimandphase[str(stim)][str(phase)]  = epochs_bystimandphase[str(stim)][str(phase)].average() 
# =============================================================================
#                 # Flipping the hemisphere
#                 if name == 'V2_R':
#                    for i_ind in non_central_ch:
#                        erp_bystimandphase[str(stim)][str(phase)] ._data[i_ind:i_ind+2,:] = np.flipud(erp_bystimandphase[str(stim)][str(phase)] ._data[i_ind:i_ind+2,:])
#                    else:
# 
#                          erp_bystimandphase[str(stim)][str(phase)]=   erp_bystimandphase[str(stim)][str(phase)]
# =============================================================================
                        
                for ipeak, peak in enumerate(labels):
              
                    
                    if ipeak == 0:    #P45
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                    
                    elif  ipeak == 1: #N60
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[1,0]:win_erps[1,1]]),1)
                    
                    elif  ipeak == 2: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[2,0]:win_erps[2,1]]),1)

                    elif  ipeak == 3: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[3,0]:win_erps[3,1]]),1)
                   

                    if str(erp_bystimandphase[str(stim)][str(phase)].comment) == str(''):    # To remove none arrays after selecting epochs
                        peaks_bystimandphase[str(stim)][str(phase)] = np.zeros(64) 
                       
                    else:
                        peaks[ifiles, :, ipeak, iphase, istim] = (peaks_bystimandphase[str(stim)][str(phase)] )
             
    

              
                

    adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs_bystimandphase[str(stim)][str(phase)].info , 'eeg')
    clusters, mask, pvals = permutation_cluster_each_intensity(peaks, adjacency_mat, thresholds, i_intensity)         
    nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)    
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters)):
        allclusters[:,p] = clusters[p][0]
    # set all other t values to 0 to focus on clusters
    allclusters[mask==False] = 0
    ch_names = epochs.ch_names
    # this is putting the 5-dim data structure in the right format for performing the sine fits
    
    for p in range(len(clusters)):
        a[:,p] = clusters[p][0]
        
    #combine labels 2 and 3, they are the same component. just different cluster    
    #a_com = a[:,[0, 1, 2, 4]]
    a[a > 8] = 8
    a[a < -8] = -8
    fig = plot_topomap_peaks_second_v(a, mask, ch_names, pvals,[-5,5], epochs.info , i_intensity = 'all')
    
    
    
    # Name and indices of the EEG electrodes that are in the biggest cluster
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask[:,p] == 1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask[:,p] == 1)[0]])
        

    fig.savefig( save_folder + name +'_' +f'{ (i_intensity+1)*2}'+ '.svg')    

    return all_ch_names_biggets_cluster, pvals, a, mask, epochs.info



def plot_topomap_peaks_second_v(name, peaks, mask, ch_names, pvals, clim, pos, i_intensity):

   
    nplots =1 
    nchans, npeaks = np.shape(peaks)

    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=5)

    fig, sps = plt.subplots(nrows=nplots, ncols=npeaks, figsize=(10,6))
    plt.style.use('default')
    
    for iplot in range(nplots):
        for ipeak in range(npeaks):

            # if mask is None:
            #     psig = None
            # else:
            #     psig = np.where(mask[iplot, :, ipeak] < 0.01, True, False)

            # sps[ipeak, iplot].set_aspect('equal')

            if mask is not None:
                imask=mask[:,ipeak]
            else:
                imask = None

            im = topoplot_2d(ch_names, peaks[ :, ipeak], pos,
                                clim=clim, axes=sps[ipeak], 
                                mask=imask, maskparam=maskparam)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    if i_intensity == 'all':
        fig.suptitle('All Intensities and Phases Vs zero_' + f'{name}', fontsize = 14)
    else:
        fig.suptitle(f'{ (i_intensity+1)*2} mA', fontsize = 14)
    #fig.suptitle('All Frequencies and all phases', fontsize = 14)
# =============================================================================
#     sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n  cluster_pv = {pvals[0]}')
#     sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {pvals[1]}')
# =============================================================================
    sps[0].set_title('\n\n P1' , fontsize=14, fontweight ='bold')
    sps[1].set_title('\n\n N1' , fontsize=14, fontweight ='bold')
    sps[2].set_title('\n\n N2',  fontsize=14, fontweight ='bold')
    sps[3].set_title('\n\n P2',  fontsize=14, fontweight ='bold')
    #sps[3].set_title('\n\n P200', fontsize=14, fontweight ='bold')
    if pvals is not None:
        fig.text(0.17, 0.3, f' P = {np.round(pvals[0], 3)} ',  ha='left',fontsize=14)
        fig.text(0.34, 0.3, f' P = {np.round(pvals[1], 3)} ',  ha='left', fontsize=14)
        fig.text(0.55,  0.3, f' P = {np.round(pvals[2], 3)} ',  ha='left',fontsize=14)
        fig.text(0.75, 0.3, f' P = {np.round(pvals[3], 3)} ',  ha='left',fontsize=14)
        #fig.text(0.73, 0.3, f' P = {np.round(pvals[3], 2)} ',  ha='left', fontsize=14)
    
    #fig.text(0, 0. ,f' \n\n  TH = {thresholds[0]} \n\n  cluster_pv = {pvals_all[str(0)]}\n\n {all_ch_names_biggets_cluster[str(0)][str(0)]}\n\n {all_ch_names_biggets_cluster[str(0)][str(1)]}\n\n  ',  ha='left')
    #fig.text(0.5, 0 ,f' \n\n  TH = {thresholds[1]} \n\n  cluster_pv = {pvals_all[str(1)]}\n\n {all_ch_names_biggets_cluster[str(1)][str(0)]}\n\n {all_ch_names_biggets_cluster[str(1)][str(1)]}\n\n  ',  ha='left')
    cb.set_label('t-value', rotation = 90)

    

    plt.show()

    return fig




def plot_topomap_peaks_second_v_com(peaks, mask, ch_names, pvals, clim, pos, i_intensity):

   
    nplots =1 
    nchans, npeaks = np.shape(peaks)

    maskparam = dict(marker='X', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=5)

    fig, sps = plt.subplots(nrows=nplots, ncols=npeaks, figsize=(10,6))
    plt.style.use('default')
    
    for iplot in range(nplots):
        for ipeak in range(npeaks):

            # if mask is None:
            #     psig = None
            # else:
            #     psig = np.where(mask[iplot, :, ipeak] < 0.01, True, False)

            # sps[ipeak, iplot].set_aspect('equal')

            if mask is not None:
                imask=mask[:,ipeak]
            else:
                imask = None

            im = topoplot_2d(ch_names, peaks[ :, ipeak], pos,
                                clim=clim, axes=sps[ipeak], 
                                mask=imask, maskparam=maskparam)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    if i_intensity == 'all':
        fig.suptitle('All Intensities and Phases Vs zero', fontsize = 14)
    else:
        fig.suptitle(f'{ (i_intensity+1)*2} mA', fontsize = 14)
    #fig.suptitle('All Frequencies and all phases', fontsize = 14)
# =============================================================================
#     sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n  cluster_pv = {pvals[0]}')
#     sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {pvals[1]}')
# =============================================================================
    sps[0].set_title('\n\n P1' , fontsize=14, fontweight ='bold')
    sps[1].set_title('\n\n N1' , fontsize=14, fontweight ='bold')
    sps[2].set_title('\n\n P2', fontsize=14, fontweight ='bold')
    #sps[3].set_title('\n\n P200', fontsize=14, fontweight ='bold')
    if pvals is not None:
        fig.text(0.18, 0.3, f' P = {np.round(pvals[0], 3)} ',  ha='left',fontsize=14)
        fig.text(0.45, 0.3, f' P = {np.round(pvals[1], 3)} ',  ha='left', fontsize=14)
        fig.text(0.7, 0.3, f' P = {np.round(pvals[2], 3)} ',  ha='left',fontsize=14)
        #fig.text(0.73, 0.3, f' P = {np.round(pvals[3], 2)} ',  ha='left', fontsize=14)
    
    #fig.text(0, 0. ,f' \n\n  TH = {thresholds[0]} \n\n  cluster_pv = {pvals_all[str(0)]}\n\n {all_ch_names_biggets_cluster[str(0)][str(0)]}\n\n {all_ch_names_biggets_cluster[str(0)][str(1)]}\n\n  ',  ha='left')
    #fig.text(0.5, 0 ,f' \n\n  TH = {thresholds[1]} \n\n  cluster_pv = {pvals_all[str(1)]}\n\n {all_ch_names_biggets_cluster[str(1)][str(0)]}\n\n {all_ch_names_biggets_cluster[str(1)][str(1)]}\n\n  ',  ha='left')
    cb.set_label('t-value', rotation = 90)

    

    plt.show()

    return fig



def topoplot_2d (ch_names, ch_attribute, pos, clim=None, axes=None, mask=None, maskparam=None):
    
    """
    Function to plot the EEG channels in a 2d topographical plot by color coding 
    a certain attribute of the channels (such as PSD, channel specific r-squared).
    Draws headplot and color fields.
    Parameters
    ----------
    ch_names : String of channel names to plot.
    ch_attribute : vector of values to be color coded, with the same length as the channel, numerical.
    clim : 2-element sequence with minimal and maximal value of the color range.
           The default is None.
           
    Returns
    -------
    None.
    This function is a modified version of viz.py (mkeute, github)
    """    

    import mne
    # get standard layout with over 300 channels
    #layout = mne.channels.read_layout('EEG1005')
    
    # select the channel positions with the specified channel names
    # channel positions need to be transposed so that they fit into the headplot
# =============================================================================
#     pos = (np.asanyarray([layout.pos[layout.names.index(ch)] for ch in ch_names])
#            [:, 0:2] - 0.5) / 5
#     
# =============================================================================
    if maskparam == None:
        maskparam = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                    linewidth=0, markersize=3) #default in mne
    if clim == None:
        im = mne.viz.plot_topomap(ch_attribute, 
                                  pos, 
                                  ch_type='eeg',
                                  sensors=True,
                                  contours=5,
                                  cmap = 'RdBu_r',
                                  axes=axes,
                                  outlines = "head", 
                                  mask=mask,
                                  mask_params=maskparam,
                                  vlim = (clim[0], clim[1]),
                                  sphere=(0.00, 0.00, 0.00, 0.11),
                                  extrapolate = 'head')
    else:
        im = mne.viz.plot_topomap(ch_attribute, 
                                  pos, 
                                  ch_type='eeg',
                                  sensors=True,
                                  contours=5,
                                  cmap = 'RdBu_r',
                                  axes=axes,
                                  outlines = "head", 
                                  mask=mask,
                                  mask_params=maskparam,
                                  vlim = (clim[0], clim[1]),
                                  sphere=(0.00, 0.00, 0.00, 0.11),
                                  extrapolate = 'head')
    return im





def chan_flip(chans):
    """
    tales a list of channels and flips the order of channels
    such that right hepishphere becames left and the left one becomes right

    Parameters
    ----------
    chans : list of string that contains channels names
        

    Returns
    -------
    chans :  flipped order channel list.

    """
        
    for i in range(len(chans)):
        if chans[i][-1] !='z': #ignore central channels
            if (int(chans[i][-1])) == 0: #expectional case for channels with two digits e.g. FT10, PT10
                chans[i] = chans[i][0:-1]
                chans[i] = chans[i].replace(chans[i][-1], '9')
    
    
                
            elif (int(chans[i][-1]) % 2) == 0: # check if the last number is even
                #all_channels_flipped.append (+str(int((all_channels[i][-1] ))-1))
                l_ch = str(int((chans[i][-1] ))-1)
                chans[i] = chans[i].replace(chans[i][-1], l_ch)
            else:
                #all_channels_flipped.append (str(int((all_channels[i][-1] ))+1))
                l_ch = str(int((chans[i][-1] ))+1)
                chans[i] = chans[i].replace(chans[i][-1], l_ch)
    return chans
            




def cosinus(x, amp, phi):
    return amp * np.cos(x + phi)

def unif(x, offset):
    return offset

def do_one_perm(model, params, y,x):
    resultperm = model.fit(y, params, x=x)
    return resultperm.best_values['amp']



def do_cosine_fit_justphase_ll(erp_amplitude, phase_bin_means, freq_band, labels, subjects, perm = True):

    



    """ 
    Inputs: 
        
    erp_amplitude: is a dictionary file of two ERPs of each target frequency and phase.
                   This variable is calculated by first averaging over the channels within 
                   the chosen cluster and then averaging over epochs in the main script. 
                   Z scoring happens inside this function. I have one value for each ERP,
                   target freq and target phase and I do z scoring for each ERP, target freq
                   within the phases.
                                        ____________                  
                                       \  ____0°    \      
                    ______ 4Hz ________\ |    .     \       
        ______ ERP1|______ 8Hz         \ |    .     \     
       |           |         .         \ |____315°  \   
       |           |         .         \____________\                     
ERPs               |______40Hz               \
       |                                     \                                                       
       |                                     \
       |                                     \                             
       |______ ERP2                          \
                                         z scoring
                                             \
                                             \
                                             \
                                       cosine fitting"""     


    
    cosinefit = {}
    amplitudes_cosine = np.zeros( len(labels))
    pvalues_cosine = np.zeros(len(labels))
    surrogate_mean =  np.zeros(len(labels))
    
    
    x = np.radians(np.array([0, 45, 90, 135, 180, 225, 270, 315]))
    #x = phase_bin_means
    
    for i in range(len(erp_amplitude)):
        cosinefit[str(i)] = {}

        if subjects == 'individual':
            y = zscore(list(erp_amplitude[str(i)].values()))
        else:
            y = erp_amplitude[str(i)]
# =============================================================================
#             if(math.isnan(((erp_amplitude[str(0)][str(8)].values()))[0]) == True):
#                 break
#            
# =============================================================================
        cosinefit[str(i)] = []
        fits = []
        for phase_start in [-np.pi/2, 0, np.pi/2]:   
    
            amp_start = np.sqrt(np.mean(y**2))
            model = lmfit.Model(cosinus)
            params = model.make_params()
    
            params["amp"].set(value=amp_start, min=0, max=np.ptp(y)/2)
            params["phi"].set(value=phase_start, min=-np.pi, max=np.pi)
            data = {ph: np.mean(y[x == ph]) for ph in np.unique(x)}
            #data = y
            fits.append(model.fit(y, params, x=x))
            
        result = fits[np.argmin([f.aic for f in fits])]
        
        if perm:
            model = lmfit.Model(cosinus)
            params = result.params
            dataperm = []
        
            # use all possible combinations of the 8 phase bins to determine p.
            # Take out the first combination because it is the original
            all_perms = list(itertools.permutations(x))
            del all_perms[0]
        
            for iper in tqdm(range(len(all_perms))):
                x_shuffled = all_perms[iper]
                dataperm.append([model,params, y, x_shuffled])
        
            with Pool(4) as p:
                surrogate = p.starmap(do_one_perm, dataperm)
        else: 
            surrogate = [np.nan]
            
        
        nullmodel = lmfit.Model(unif)
        params = nullmodel.make_params()
        params["offset"].set(value=np.mean(y), min=min(y), max=max(y))
        nullfit = nullmodel.fit(y, params, x=x)
        surrogate = np.array(surrogate)
        surrogate = surrogate[np.invert(np.isnan(surrogate))]
        
        cosinefit[str(i)].append( { 'Model': 'cosinus', 
                                'Fit': result,
                                'data': data, 
                                'amp': result.best_values['amp'], 
                                'surrogate': surrogate, 
                                'p':[np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0], 
                                'std':[np.nan if perm == False else np.std(surrogate)][0], 
                                'nullmodel':nullfit,
                                })
        
        amplitudes_cosine[ i] = result.best_values['amp']
        pvalues_cosine[i] = [np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0] 
        surrogate_mean[i] = np.mean(surrogate)
    
    
    return cosinefit, amplitudes_cosine, pvalues_cosine, surrogate_mean





def do_cosine_fit_ll(erp_amplitude, phase_bin_means, freq_band, labels, subjects, perm = True):

    



    """ 
    Inputs: 
        
    erp_amplitude: is a dictionary file of two ERPs of each target frequency and phase.
                   This variable is calculated by first averaging over the channels within 
                   the chosen cluster and then averaging over epochs in the main script. 
                   Z scoring happens inside this function. I have one value for each ERP,
                   target freq and target phase and I do z scoring for each ERP, target freq
                   within the phases.
                                        ____________                  
                                       \  ____0°    \      
                    ______ 4Hz ________\ |    .     \       
        ______ ERP1|______ 8Hz         \ |    .     \     
       |           |         .         \ |____315°  \   
       |           |         .         \____________\                     
ERPs               |______40Hz               \
       |                                     \                                                       
       |                                     \
       |                                     \                             
       |______ ERP2                          \
                                         z scoring
                                             \
                                             \
                                             \
                                       cosine fitting"""     


    
    cosinefit = {}
    amplitudes_cosine = np.zeros([len(freq_band), len(labels)])
    pvalues_cosine = np.zeros([len(freq_band), len(labels)])

    
    
    x = np.radians(np.array([0, 45, 90, 135, 180, 225, 270, 315]))
    #x = phase_bin_means
    
    for i in range(len(erp_amplitude)):
        cosinefit[str(i)] = {}
        for jf, f in enumerate(freq_band):    
            print('cosine fits for intensity {}'.format(f))
            if subjects == 'individual':
                y = zscore(list(erp_amplitude[str(i)][str(f)].values()))
            else:
                y = erp_amplitude[str(i)][str(f)]
# =============================================================================
#             if(math.isnan(((erp_amplitude[str(0)][str(8)].values()))[0]) == True):
#                 break
#            
# =============================================================================
            cosinefit[str(i)][str(f)] = []
            fits = []
            for phase_start in [-np.pi/2, 0, np.pi/2]:   
        
                amp_start = np.sqrt(np.mean(y**2))
                model = lmfit.Model(cosinus)
                params = model.make_params()
        
                params["amp"].set(value=amp_start, min=0, max=np.ptp(y)/2)
                params["phi"].set(value=phase_start, min=-np.pi, max=np.pi)
                data = {ph: np.mean(y[x == ph]) for ph in np.unique(x)}
                #data = y
                fits.append(model.fit(y, params, x=x))
                
            result = fits[np.argmin([f.aic for f in fits])]
            
            if perm:
                model = lmfit.Model(cosinus)
                params = result.params
                dataperm = []
            
                # use all possible combinations of the 8 phase bins to determine p.
                # Take out the first combination because it is the original
                all_perms = list(itertools.permutations(x))
                del all_perms[0]
            
                for iper in tqdm(range(len(all_perms))):
                    x_shuffled = all_perms[iper]
                    dataperm.append([model,params, y, x_shuffled])
            
                with Pool(4) as p:
                    surrogate = p.starmap(do_one_perm, dataperm)
            else: 
                surrogate = [np.nan]
                
            
            nullmodel = lmfit.Model(unif)
            params = nullmodel.make_params()
            params["offset"].set(value=np.mean(y), min=min(y), max=max(y))
            nullfit = nullmodel.fit(y, params, x=x)
            surrogate = np.array(surrogate)
            surrogate = surrogate[np.invert(np.isnan(surrogate))]
            
            cosinefit[str(i)][str(f)].append( { 'Model': 'cosinus', 
                                    'Frequency': f, 
                                    'Fit': result,
                                    'data': data, 
                                    'amp': result.best_values['amp'], 
                                    'surrogate': surrogate, 
                                    'p':[np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0], 
                                    'std':[np.nan if perm == False else np.std(surrogate)][0], 
                                    'nullmodel':nullfit,
                                    })
            
            amplitudes_cosine[jf, i] = result.best_values['amp']
            pvalues_cosine[jf, i] = [np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0] 
    
    
    
    return cosinefit, amplitudes_cosine, pvalues_cosine





def cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, ERP_indexs, exdir_epoch):
        
    stim_int = np.arange(2, 18, 2)
    
    files = list(Path(exdir_epoch).glob('*epo.fif*'))
    all_sub_evoked = []
    for i_sub, f in enumerate(files):
    
    
        
        # Extracting ERP amplitude for frequency and phases according to bipolar channel.
            
        # Subj_path is added to exdir, so the EEG epoch files and bipolar  channels are selected from the same subject. 
        epochs_eeg = mne.read_epochs(f, preload=True).copy().pick_types(eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_eeg_amp_mod = epochs_eeg._data[:,:,1001:] - epochs_eeg._data[:,:,0:1000]
        # making mne epoch structure
        epochs_eeg = mne.EpochsArray(data = epochs_eeg_amp_mod,  info = epochs_eeg.info, events = epochs_eeg.events, event_id = epochs_eeg.event_id, on_missing='ignore')
     
        
    
        epochs_byfreqandphase = {} 
        erp_amplitude_ll = {}
        ERP_byfreqandphase = {}
        evoked = {}
        evoked_z = {}
        
        for i_ch, ch in enumerate(ERP_indexs):
            epochs_byfreqandphase[str(i_ch)] = {} 
            erp_amplitude_ll[str(i_ch)] = {}
            ERP_byfreqandphase[str(i_ch)] = {}
            evoked[str(i_ch)] = {}
            evoked_z[str(i_ch)] = {}
            for istim, stim in enumerate(stim_int):
                epochs_byfreqandphase[str(i_ch)][str(stim)] = {}
                ERP_byfreqandphase[str(i_ch)][str(stim)] = {}
                evoked[str(i_ch)][str(stim)] = {}
                evoked_z[str(i_ch)][str(stim)] = {}
                for phase in np.arange(0,360,45):
                    sel_idx = Select_Epochs_intensity_phase(epochs_eeg, stim, phase) # Selecting lucky loop labels
                    epochs_byfreqandphase[str(i_ch)][str(stim)][str(phase)] = epochs_eeg[sel_idx]
                    
                    if i_ch == 0:   #P1
                        ERP_byfreqandphase[str(i_ch)][str(stim)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(stim)][str(phase)]._data[:, ch, win_erps[0,0]: win_erps[0,1]], axis=0)
                        evoked[str(i_ch)][str(stim)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(stim)][str(phase)], axis = 1))
                    
                    elif i_ch == 1: #N1
                        ERP_byfreqandphase[str(i_ch)][str(stim)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(stim)][str(phase)]._data[:, ch, win_erps[1,0]: win_erps[1,1]], axis=0)
                        evoked[str(i_ch)][str(stim)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(stim)][str(phase)], axis = 1))
                     
                    elif i_ch == 2: #N2
                        ERP_byfreqandphase[str(i_ch)][str(stim)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(stim)][str(phase)]._data[:, ch, win_erps[2,0]: win_erps[2,1]], axis=0)
                        evoked[str(i_ch)][str(stim)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(stim)][str(phase)], axis = 1))
                    elif i_ch == 3: #P2
                        ERP_byfreqandphase[str(i_ch)][str(stim)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(stim)][str(phase)]._data[:, ch, win_erps[3,0]: win_erps[3,1]], axis=0)
                        evoked[str(i_ch)][str(stim)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(stim)][str(phase)], axis = 1))
              
    
                    if str(evoked[str(i_ch)][str(stim)][str(phase)]) == 'nan':
                        evoked[str(i_ch)][str(stim)][str(phase)] = 0 # removing 'nan' objects
                evoked_z[str(i_ch)][str(stim)] = zscore(list(evoked[str(i_ch)][str(stim)].values()))        
        all_sub_evoked.append(evoked_z) 
    
    
    
    
    
    
    # calculating z-scores per subject, and target frequency to render ERPs
    all_sub_evoked_df = pd.DataFrame(all_sub_evoked)       
    all_evoked_freq = {}
    for i_erp, i in enumerate(ERP_indexs):
        all_evoked_freq[str(i_erp)] = {}
        for stim in np.arange(2, 18, 2):
            avg = []
            for i_sub in range(len(all_sub_evoked)):
                avg.append(all_sub_evoked_df[str(i_erp)][i_sub][str(stim)])
            all_evoked_freq[str(i_erp)][str(stim)] = np.mean(avg, axis = 0)
    

    cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll = do_cosine_fit_ll(all_evoked_freq, np.arange(0,360,45), np.arange(2,18,2), labels, subjects = 'group' , perm = True)


    return cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll



def cosine_fit_phase_overlapping_chs(win_erps, labels, ERP_indexs, exdir_epoch):


    files = list(Path(exdir_epoch).glob('*epo.fif*'))
    all_sub_evoked = []
    for i_sub, f in enumerate(files):
    
    
        
        # Extracting ERP amplitude for frequency and phases according to bipolar channel.
            
        # Subj_path is added to exdir, so the EEG epoch files and bipolar  channels are selected from the same subject. 
        epochs_eeg = mne.read_epochs(f, preload=True).copy().pick_types(eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_eeg_amp_mod = epochs_eeg._data[:,:,1001:] - epochs_eeg._data[:,:,0:1000]
        # making mne epoch structure
        epochs_eeg = mne.EpochsArray(data = epochs_eeg_amp_mod,  info = epochs_eeg.info, events = epochs_eeg.events, event_id = epochs_eeg.event_id, on_missing='ignore')
     
        
    
        epochs_byfreqandphase = {} 
        erp_amplitude_ll = {}
        ERP_byfreqandphase = {}
        evoked = {}
        evoked_z = {}
        
        for i_ch, ch in enumerate(ERP_indexs):
            epochs_byfreqandphase[str(i_ch)] = {} 
            erp_amplitude_ll[str(i_ch)] = {}
            ERP_byfreqandphase[str(i_ch)] = {}
            evoked[str(i_ch)] = {}
            evoked_z[str(i_ch)] = {}
         
            for phase in np.arange(0,360,45):
                sel_idx = Select_Epochs_phase(epochs_eeg, phase) # Selecting lucky loop labels
                epochs_byfreqandphase[str(i_ch)][str(phase)] = epochs_eeg[sel_idx]
                
                if i_ch == 0:   #P1
                    ERP_byfreqandphase[str(i_ch)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(phase)]._data[:, ch, win_erps[0,0]: win_erps[0,1]], axis=0)
                    evoked[str(i_ch)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(phase)], axis = 1))
                
                elif i_ch == 1: #N1
                    ERP_byfreqandphase[str(i_ch)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(phase)]._data[:, ch, win_erps[1,0]: win_erps[1,1]], axis=0)
                    evoked[str(i_ch)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(phase)], axis = 1))
                 
                elif i_ch == 2: #N2
                    ERP_byfreqandphase[str(i_ch)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(phase)]._data[:, ch, win_erps[2,0]: win_erps[2,1]], axis=0)
                    evoked[str(i_ch)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(phase)], axis = 1))
                    
                elif i_ch == 3: #P2
                    ERP_byfreqandphase[str(i_ch)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(phase)]._data[:, ch, win_erps[3,0]: win_erps[3,1]], axis=0)
                    evoked[str(i_ch)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(phase)], axis = 1))

       
                       
    
                if str(evoked[str(i_ch)][str(phase)]) == 'nan':
                    evoked[str(i_ch)][str(phase)] = 0 # removing 'nan' objects
            evoked_z[str(i_ch)] = zscore(list(evoked[str(i_ch)].values()))        
        all_sub_evoked.append(evoked_z) 
    
    
    
    
    
    
    # calculating z-scores per subject, and target frequency to render ERPs
    all_sub_evoked_df = pd.DataFrame(all_sub_evoked)       
    all_evoked_freq = {}
    for i_erp, i in enumerate(ERP_indexs):
        avg = []
        all_evoked_freq[str(i_erp)] = {}
    
        for i_sub in range(len(all_sub_evoked)):
            avg.append(all_sub_evoked_df[str(i_erp)][i_sub])
        all_evoked_freq[str(i_erp)] = np.average(avg, axis = 0)
    

    cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll, surrogate = do_cosine_fit_justphase_ll(all_evoked_freq, np.arange(0,360,45), np.arange(2,18,2), labels, subjects = 'group' , perm = True)

    return cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll, surrogate

def cosine_fit_phase_ind_overlapping_chs(win_erps, labels, ERP_indexs, exdir_epoch):
    cosine_fit_all_subjects_LL = []
    amplitudes_cosines_all_subjects_LL = []

    files = list(Path(exdir_epoch).glob('*epo.fif*'))

    for i_sub, f in enumerate(files):
    
    
        
        # Extracting ERP amplitude for frequency and phases according to bipolar channel.
            
        # Subj_path is added to exdir, so the EEG epoch files and bipolar  channels are selected from the same subject. 
        epochs_eeg = mne.read_epochs(f, preload=True).copy().pick_types(eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_eeg_amp_mod = epochs_eeg._data[:,:,1001:] - epochs_eeg._data[:,:,0:1000]
        # making mne epoch structure
        epochs_eeg = mne.EpochsArray(data = epochs_eeg_amp_mod,  info = epochs_eeg.info, events = epochs_eeg.events, event_id = epochs_eeg.event_id, on_missing='ignore')
     
        
    
        epochs_byfreqandphase = {} 
        erp_amplitude_ll = {}
        ERP_byfreqandphase = {}
        evoked = {}
        evoked_z = {}
        
        for i_ch, ch in enumerate(ERP_indexs):
            epochs_byfreqandphase[str(i_ch)] = {} 
            erp_amplitude_ll[str(i_ch)] = {}
            ERP_byfreqandphase[str(i_ch)] = {}
            evoked[str(i_ch)] = {}
            evoked_z[str(i_ch)] = {}
         
            for phase in np.arange(0,360,45):
                sel_idx = Select_Epochs_phase(epochs_eeg, phase) # Selecting lucky loop labels
                epochs_byfreqandphase[str(i_ch)][str(phase)] = epochs_eeg[sel_idx]
                
                if i_ch == 0:   #P1
                    ERP_byfreqandphase[str(i_ch)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(phase)]._data[:, ch, win_erps[0,0]: win_erps[0,1]], axis=0)
                    evoked[str(i_ch)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(phase)], axis = 1))
                
                elif i_ch == 1: #N1
                    ERP_byfreqandphase[str(i_ch)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(phase)]._data[:, ch, win_erps[1,0]: win_erps[1,1]], axis=0)
                    evoked[str(i_ch)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(phase)], axis = 1))
                 
                elif i_ch == 2: #N2
                    ERP_byfreqandphase[str(i_ch)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(phase)]._data[:, ch, win_erps[2,0]: win_erps[2,1]], axis=0)
                    evoked[str(i_ch)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(phase)], axis = 1))
                    
                elif i_ch == 3: #P2
                    ERP_byfreqandphase[str(i_ch)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(phase)]._data[:, ch, win_erps[3,0]: win_erps[3,1]], axis=0)
                    evoked[str(i_ch)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(phase)], axis = 1))

       
                       
    
                if str(evoked[str(i_ch)][str(phase)]) == 'nan':
                    evoked[str(i_ch)][str(phase)] = 0 # removing 'nan' objects
                    #evoked_z[str(i_ch)] = zscore(list(evoked[str(i_ch)].values()))        
    

        cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll, surrogate = do_cosine_fit_justphase_ll(evoked, np.arange(0,360,45), np.arange(2,18,2), labels, subjects = 'individual' , perm = True)


                    
        amplitudes_cosines_all_subjects_LL.append(amplitudes_cosine_ll)
        cosine_fit_all_subjects_LL.append(cosinefit_ll)


    return cosine_fit_all_subjects_LL, amplitudes_cosines_all_subjects_LL

def Circ_corr(cosinefit_ll, labels):
    fig, ax =  plt.subplots(1,len(labels))
    
    phi_dict = {}
    intensity_band = np.arange(2, 18, 2)
    phi_array_deg = np.zeros([len(intensity_band), len(labels)])
    amp_array_all = np.zeros([len(intensity_band), len(labels)])
    for i in range(len(labels)):
        amp = {}
        phi = {}
        for ji, intensity in enumerate(intensity_band):   
                
            amp[str(intensity)] = cosinefit_ll[str(i)][str(intensity)][0]['amp']  
            phi[str(intensity)] = cosinefit_ll[str(i)][str(intensity)][0]['Fit'].best_values['phi']
        
    
        
                
        amp_df = pd.DataFrame({'2mA': amp[str(2)], '4mA': amp[str(4)], '6mA': amp[str(6)], '8mA': amp[str(8)], \
                               '10mA': amp[str(10)], '12mA': amp[str(12)], '14mA': amp[str(14)], '16mA': amp[str(16)]}, index=['amp'])   
        amp_df_array = amp_df.to_numpy()
    
        phi_df = pd.DataFrame({'2mA': phi[str(2)], '4mA': phi[str(4)], '6mA': phi[str(6)], '8mA': phi[str(8)], \
                                    '10mA': phi[str(10)], '12mA': phi[str(12)], '14mA': phi[str(14)], '16mA': phi[str(16)]}, index=['phi']) 
        phi_array = phi_df.to_numpy()
    
        phi_dict[str(i)] = phi_array
        amp_array_all[:,i] =  amp_df_array
        phi_array_deg[:,i] =  np.degrees(phi_array)
        for j, j2 in enumerate(intensity_band):

            phi_array_deg[j,i] =  phi_array_deg[j,i] 
            
         
         
    
        
    
        cor, ci = pycircstat.corrcc(np.array(intensity_band), phi_array_deg[:,i], ci=True)
        cor = np.abs(cor)
        rval = str(np.round(cor,4))
        tval = (cor*(np.sqrt(len(np.array(intensity_band)-2)))/(np.sqrt(1-cor**2)))
        pval = str(np.round(1-stats.t.cdf(np.abs(tval),len(np.array(intensity_band))-1),3))
        # plot scatter
     
    
        im = ax[i].scatter(phi_array_deg[:,i], intensity_band, c= amp_df_array)    
        if i==0:
            erp_num = 'First'
        else:
            erp_num = 'Second'
        ax[i].title.set_text(f'{erp_num} ERP, r = {rval}, p = {pval}' )
        clb = fig.colorbar(im, ax=ax[i])    
        clb.ax.set_title('Strength of MD')
        fig.suptitle('Group Average, Real-Time')
        ax[i].set_xlim([0, 400])
        ax[i].set_xlabel('Optimal phases (deg)')
        ax[i].set_ylabel('Intensity (mA)')
        ax[i].set_xlim(left=-10)
        plt.show()
    return(phi_array_deg)





def amp_p_chi(cosinefit_ll, labels, phi_array_deg, save_folder):
    intensity_band = np.arange(2, 18, 2)
    mod_depth = {}
    p_val = {}
    red_chi = {}
    for i in range(len(labels)):
        mod_depth[str(i)] = {} 
        p_val[str(i)] = {}
        red_chi[str(i)] = {}
        for ji, intensity in enumerate(intensity_band):   
           mod_depth[str(i)][str(intensity)] = cosinefit_ll[str(i)][str(intensity)][0]['amp']
           p_val[str(i)][str(intensity)] = cosinefit_ll[str(i)][str(intensity)][0]['p']
           red_chi[str(i)][str(intensity)]  = cosinefit_ll[str(i)][str(intensity)][0]['Fit'].redchi
           
    p_val_df = pd.DataFrame(p_val)   
    amp_df = pd.DataFrame(mod_depth)
    red_chi_df = pd.DataFrame(red_chi)
    if len(labels)==4:
        amp_df_rename = amp_df.rename(columns={'0': labels[0], '1': labels[1], '2': labels[2], '3': labels[3]})
        p_val_df_rename = p_val_df.rename(columns={'0': labels[0], '1': labels[1], '2': labels[2], '3': labels[3]})
        red_chi_df_rename  = red_chi_df.rename(columns={'0': labels[0], '1': labels[1], '2': labels[2], '3': labels[3]})
    else:
        amp_df_rename = amp_df.rename(columns={'0': labels[0], '1': labels[1]})
        p_val_df_rename = p_val_df.rename(columns={'0': labels[0], '1': labels[1]})
        red_chi_df_rename  = red_chi_df.rename(columns={'0': labels[0], '1': labels[1]})
        
        
        
    
    for i in np.arange(len(amp_df)):
        amp_df_rename = amp_df_rename.rename(index = {f'{amp_df.index[i]}' : f'{amp_df.index[i]} mA'})
    amp_df_r = amp_df_rename.T
    
    
    plt.style.use('default')
    
    # Plot modulation depth in bar format
    fig, ax = plt.subplots(1, 1)
    amp_df_r.plot(kind="bar", alpha=0.75, rot=0,  colormap = 'viridis_r', title = 'Strength of Modulation Depth,  Group Average', ax= ax).legend(loc = 'lower right')
    ax.set_ylim(bottom=-.010, top=1)
    
    
    
    
    # plot p values in a table format
    fig, ax = plt.subplots(figsize=(5,7))
    df = p_val_df_rename
    df.index.name = "Intensity"
    
    
    
    threshold = 0.05

    # Find indices of values below the threshold
    below_threshold_indices = df[df < threshold].stack().index.tolist()
    
    # Generate cell colors
    cell_colors = []
    for i, row in df.iterrows():
        row_colors = []
        for col in df.columns:
            if (i, col) in below_threshold_indices:
                row_colors.append("lightpink")  # Light red for values below threshold
            else:
                row_colors.append("w")  # Light green for values above or equal to threshold
        cell_colors.append(row_colors)
        
    
    
    
    
    table = ax.table(cellText=np.round(df.values, 3), rowLabels=df.index + 'mA', cellLoc='center',
                     cellColours=cell_colors, colLabels=df.columns, loc='center',
                     colWidths= [0.15]*(len(df.columns)))
    
    w, h = table[0,1].get_width(), table[0,1].get_height()
    table.add_cell(0, -1, w,h, text=df.index.name)
    ax.grid(False)
    table.scale(1,1.2)
    plt.title('p values')
    plt.show()
    fig.savefig(save_folder +'P_value.svg')
    
    
    
    # plot reduced chi squared in atable format
    fig, ax = plt.subplots(figsize=(5,7))
    df = red_chi_df_rename
    df.index.name = "Intensity"
    table = ax.table(cellText=np.round(df.values, 3), rowLabels=df.index + 'mA', cellLoc='center',
                     colColours=['gainsboro'] * len(df), colLabels=df.columns, loc='center',
                     colWidths= [0.15]*(len(df.columns)))
    
    w, h = table[0,1].get_width(), table[0,1].get_height()
    table.add_cell(0, -1, w,h, text=df.index.name)
    ax.grid(False)
    table.scale(1,1.2)
    plt.title('Reduced chi-squared')
    plt.show()
    


    
    # Plot optimal phases
    # Here we need phase lag, not phase lead. So if a value is positive we need to diffrentiate is from 360. 
    
    phi_array_deg_correct = np.zeros([len(intensity_band), len(labels)])

    
    for i_erp in np.arange((phi_array_deg.shape[1])):
        for i in np.arange(len(phi_array_deg)):

            # It's about phase lead and lag compared to the cosine with no phase shift
            # the phi that I get from fitted cosine, actually shows me how many degrees it leads or lags a cosine with no phase shift
            # look at "best_fit_plot" to understand it better
            if phi_array_deg[i][i_erp] < 0:
               #phi_array_deg_correct[i][i_erp] = 360 + phi_array_deg[i][i_erp]
               phi_array_deg_correct[i][i_erp] = abs(phi_array_deg[i][i_erp])
            else:

                 phi_array_deg_correct[i][i_erp] =  360 - phi_array_deg[i][i_erp]
                
    
    # It's about phase lead and lag compared to the cosine with no phase shift
    df = pd.DataFrame(phi_array_deg_correct)
    if len(labels)==4:
        df = df.rename(columns={0: labels[0], 1: labels[1], 2: labels[2], 3: labels[3]})
    else:
         df = df.rename(columns={0: labels[0], 1: labels[1]})
    for i in np.arange(len(phi_array_deg_correct)):
        df = df.rename(index = {df.index[i] : f'{((df.index[i] + 1)*2)} '})
    fig, ax = plt.subplots(figsize=(7,10))    
    df.index.name = "Intensity"
    table = ax.table(cellText=np.round(df.values, 3), rowLabels=df.index + 'mA', cellLoc='center',
                     colColours=['gainsboro'] * len(df), colLabels=df.columns, loc='center',
                     colWidths= [0.15]*(len(df.columns)))
    
    w, h = table[0,1].get_width(), table[0,1].get_height()
    table.add_cell(0, -1, w,h, text=df.index.name)
    ax.grid(False)
    table.scale(1,1.3)
    plt.title('Optimal Phases')
    plt.show()
    fig.savefig(save_folder +'Optimal_phases.svg')

    return(pd.DataFrame.to_numpy(p_val_df_rename), phi_array_deg_correct, amp_df_r)










def best_fit_plot(cosinefit_ll, labels, phi_array_deg_correct, save_folder, title, y_limit):
    
    X= np.array([0, 45, 90, 135, 180, 225, 270, 315])
    intensity_band = np.arange(2, 18, 2)
    # Plotting data and interpolated fitted cosine 
    Y = {}
    Y_shifted ={}
    best_fit = {}
    best_fit_shifted = {}
    for i in range(len(labels)):
        Y[str(i)] = {}
        Y_shifted[str(i)] ={}
        best_fit[str(i)] ={}
        best_fit_shifted[str(i)] = {}
        for ji, intensity in enumerate(intensity_band):   
            Y[str(i)][str(ji)] = (np.array(list(cosinefit_ll[str(i)][str(intensity)][0]['data'].values())))
            #Y[str(i)][str(jf)] = Y[str(i)][str(jf)] - np.mean(Y[str(i)][str(jf)])        
            best_fit[str(i)][str(ji)] = cosinefit_ll[str(i)][str(intensity)][0]['Fit'].best_fit
            
            #find the maximum value (corresponsing to optimal phase), shift it as optimal phase value
            max_best_fit = np.argmax(best_fit[str(i)][str(ji)])
            max_Y =  np.argmax(Y[str(i)][str(ji)])
            Y_shifted[str(i)][str(ji)] = np.roll(Y[str(i)][str(ji)], max_Y)
            best_fit_shifted[str(i)][str(ji)] =  np.roll(best_fit[str(i)][str(ji)], max_best_fit)
    
    plt.style.use('default')  
    # Plotting data and interpolated fitted cosine 
    xi = range(len(X))
    x_new = np.linspace(0, 7, num=40)
    cols = [format(col) for col in np.arange(0,8,1)]
    

    for i_labels, name_labels in enumerate(labels):
         
        
    
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,8))
        fig.suptitle(f'{name_labels}', fontsize=16, fontweight ='bold')
        
        for ax, col in zip(axes[0], cols):
            ax.set_title(f'{(int(col)+1)*2}mA', fontsize=14, fontweight ='bold')
            ax.scatter(xi, Y[str(i_labels)][col], color='k')
            ax.set_ylim(y_limit)
            ax.set_xlim(-1, 8)
            f2 = interp1d(xi, best_fit[str(i_labels)][col], kind='cubic')
            ax.plot(x_new,  f2(x_new), 'r')
            ax.set_xticks(xi, [round(i) for i in np.arange(0,360, 45)], fontsize=8)
        
        
        
        for ax, col in zip(axes[1], cols):
            ax.set_title(f'{(int(col)+5)*2}mA', fontsize=14, fontweight ='bold')
            col = f'{int(col)+4 }'
            ax.scatter(xi, Y[str(i_labels)][col], color='k')
            ax.set_ylim(y_limit)
            ax.set_xlim(-1, 8)
            f2 = interp1d(xi, best_fit[str(i_labels)][col], kind='cubic')
            ax.plot(x_new,  f2(x_new), 'r')
            ax.set_xticks(xi, [round(i) for i in np.arange(0,360, 45)], fontsize=8)
        
        plt.show()
        fig.savefig(save_folder + f'{title}_' + f'{name_labels}'+'_cos_fit' +'.svg')

    return 



def fig_2c_scatter_sub_plot(cosinefit_ll, labels, p_val, save_folder):
    plt.style.use('default')
    intensity_band = np.arange(2, 18, 2)
    xi = range(len(np.array(intensity_band)))
    fig = fig, ax = plt.subplots(1, len(labels), figsize=(20, 4))
    # Fig 2.c
    for i_labels, name_labels in enumerate(labels):
         mod_depth = {}
         surr = {}
         for ji, intensity in enumerate(intensity_band): 

            mod_depth[str(intensity)] = cosinefit_ll[str(i_labels)][str(intensity)][0]['amp']
            surr[str(intensity)] = np.mean(cosinefit_ll[str(i_labels)][str(intensity)][0]['surrogate'])
    
    
    
    
            
         ax[i_labels].scatter(xi, (np.array(list(mod_depth.values()))),  c = 'k', label='Real data')
         ax[i_labels].plot(xi, (np.array(list(mod_depth.values()))),  c = 'k', alpha=0.1)
    
         ax[i_labels].scatter(xi, ((np.array(list(surr.values())))),  c = 'r', label='Surrogate')
         ax[i_labels].plot(xi, ((np.array(list(surr.values())))),  c = 'r', alpha= 0.1)
         ax[i_labels].set_xticks(xi, np.array(intensity_band))
         ax[i_labels].set_xlabel("Intensities (mA)", weight='bold')
         ax[i_labels].set_ylabel("Modulation depth", weight='bold')
         ax[i_labels].legend(loc='upper right')
         ax[i_labels].set_ylim(0, 1)
         ax[i_labels].set_title(f'{name_labels} Group Cosine Models', weight='bold')
         for ji, intensity in enumerate(intensity_band): 
             if p_val[ji, i_labels] < 0.05:
                ax[i_labels].plot(xi[ji],  0.8, '*', color = 'k')
         plt.show()
         fig.savefig(save_folder+'all_labels.svg')
         
    fig.tight_layout()
    return fig




def individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, ERP_indexs, exdir_epoch):
    files = list(Path(exdir_epoch).glob('*epo.fif*'))
    all_subjects_names = []
    cosine_fit_all_subjects_LL = []
    amplitudes_cosines_all_subjects_LL = []
    
    
    
    
    for i_sub, f in enumerate(files):
        subject_info = f.parts 
        all_subjects_names.append(str(subject_info[-1][0:9]))
        epochs_eeg = mne.read_epochs(f, preload=True).copy().pick_types(eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_eeg_amp_mod = epochs_eeg._data[:,:,1001:] - epochs_eeg._data[:,:,0:1000]
        epochs_eeg = mne.EpochsArray(data = epochs_eeg_amp_mod,  info = epochs_eeg.info, events = epochs_eeg.events, event_id = epochs_eeg.event_id, on_missing='ignore')
        # 4 Hz step with lucky loop labels
        evoked = {}
        erp_amplitude_ll = {}
        ERP_byfreqandphase = {}
        epochs_byfreqandphase = {} 
    
        
        for i_ch, ch in enumerate(ERP_indexs):
            evoked[str(i_ch)] = {}
            erp_amplitude_ll[str(i_ch)] = {}
            ERP_byfreqandphase[str(i_ch)] = {}
            epochs_byfreqandphase[str(i_ch)] = {} 
            for intensity in np.arange(2,18,2):
                epochs_byfreqandphase[str(i_ch)][str(intensity)] = {}
                ERP_byfreqandphase[str(i_ch)][str(intensity)] = {}
                evoked[str(i_ch)][str(intensity)] = {}
                for phase in np.arange(0,360,45):
                    sel_idx = Select_Epochs_intensity_phase(epochs_eeg, intensity, phase)
                    epochs_byfreqandphase[str(i_ch)][str(intensity)][str(phase)] = epochs_eeg[sel_idx]
                    
                    if i_ch == 0: #P1
                        ERP_byfreqandphase[str(i_ch)][str(intensity)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(intensity)][str(phase)]._data[:, ch, win_erps[0,0]: win_erps[0,1]], axis=0)
                        evoked[str(i_ch)][str(intensity)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(intensity)][str(phase)], axis = 1))
                   
                    elif i_ch == 1: #N1
                        ERP_byfreqandphase[str(i_ch)][str(intensity)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(intensity)][str(phase)]._data[:, ch, win_erps[1,0]: win_erps[1,1]], axis=0)
                        evoked[str(i_ch)][str(intensity)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(intensity)][str(phase)], axis = 1))
                     
                    elif i_ch == 2: #N2
                        ERP_byfreqandphase[str(i_ch)][str(intensity)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(intensity)][str(phase)]._data[:, ch, win_erps[2,0]: win_erps[2,1]], axis=0)
                        evoked[str(i_ch)][str(intensity)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(intensity)][str(phase)], axis = 1))
           
            
                    elif i_ch == 3: #P2
                        ERP_byfreqandphase[str(i_ch)][str(intensity)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(intensity)][str(phase)]._data[:, ch, win_erps[3,0]: win_erps[3,1]], axis=0)
                        evoked[str(i_ch)][str(intensity)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(intensity)][str(phase)], axis = 1))
 
                   
                    if str(evoked[str(i_ch)][str(intensity)][str(phase)]) == 'nan':
                        evoked[str(i_ch)][str(intensity)][str(phase)] = 0
                
     
        cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll = do_cosine_fit_ll(evoked, np.arange(0,360,45), np.arange(2,18,2), labels, subjects = 'individual' , perm = True)
        
        amplitudes_cosines_all_subjects_LL.append(amplitudes_cosine_ll)
        cosine_fit_all_subjects_LL.append(cosinefit_ll)
        
        if not (cosinefit_ll[str(0)] ):
            print(f'There are not enough epochs by freq and phase for Subject: {subject_info[-3]}')
            
    return amplitudes_cosines_all_subjects_LL, cosine_fit_all_subjects_LL, all_subjects_names


def bar_plot_avg_intensity(labels, amplitudes_cosine_avg_int, surrogate_avg, pvalues_cosine_avg_int, title, save_folder ):
    x = labels
    y = amplitudes_cosine_avg_int
    y1 = surrogate_avg
    p_val = pvalues_cosine_avg_int
    
    fig, ax = plt.subplots(figsize=(5,5))
    text_kwargs = dict(ha='center', va='bottom')
    plt.bar(x, y, color = 'k', alpha = 0.5, label='Real data')
    plt.bar(x, y1, color = 'r', alpha = 0.1, label='Surrogate')
    plt.legend(loc='upper left')
    for x, y, p in zip(x, y, p_val):
       plt.text(x, y, np.round(p,3), **text_kwargs)
    plt.show()
    plt.ylim(bottom=-.010, top=1)
    fig.savefig(save_folder + f'{title}' +'.png')
    return fig

def bar_plot_avg_intensity_ind(labels, avg_mod_depth_ind_l_st, avg_surrogate_ind_l_st, avg_phi_ind_l_st, title, save_folder ):
    x = np.arange(len(labels))
    width = 0.2 
    mod_depth = np.zeros([len(avg_mod_depth_ind_l_st), len(labels)])
    surrogate = np.zeros([len(avg_mod_depth_ind_l_st), len(labels)])
    
    for i_labels, name_labels in enumerate(labels):
        for num_sub in range(len(avg_mod_depth_ind_l_st)):  
            mod_depth[num_sub, i_labels] = avg_mod_depth_ind_l_st[str(num_sub)][str(i_labels)]
            surrogate[num_sub, i_labels]  = np.mean(avg_surrogate_ind_l_st[str(num_sub)][str(i_labels)])
       
            
       
    t_ind , p_ind = stats.ttest_ind(mod_depth, surrogate)   
    y = np.mean(mod_depth, axis=0) 
    y1 = np.mean(surrogate, axis=0)
    fig, ax = plt.subplots(figsize=(5,5))
    text_kwargs = dict(ha='center', va='bottom')

    ax.bar(x , height = np.mean(mod_depth, axis=0) , yerr= np.std(mod_depth, axis=0) , capsize=4, color ='grey', alpha = 0.5, width=width, label='Real data')
    ax.bar(x+width , height = np.mean(surrogate, axis=0) , yerr= np.std(surrogate, axis=0) , capsize=4, color ='r', alpha = 0.2, width=width, label='Surrogate data')
    for x1, y1, p in zip(x, y, p_ind):
       plt.text(x1+0.1, 1, np.round(p,3), **text_kwargs)
    ax.set_ylim(0, 1.3)
    ax.set_xticks(x+width)
    ax.set_xticklabels(labels)
    plt.legend(loc='upper left')


    return fig




def reading_cosine_function_parameters(x, labels):
    
    p = {}
    chi = {}
    phi = {}
    mag = {}
    chi_red = {}
    mod_depth = {}
    surrogate = {}
    int_band = np.arange(2, 18, 2)
    
    for num_sub in range(len(x)):
        p[str(num_sub)] = {}
        chi[str(num_sub)] = {}
        phi[str(num_sub)] = {}
        mag[str(num_sub)] = {}
        chi_red[str(num_sub)] = {}
        mod_depth[str(num_sub)] = {}
        surrogate[str(num_sub)] = {}
    
        
    
            
        for i in range(len(labels)): 
            p[str(num_sub)][str(i)]  = {}
            chi[str(num_sub)][str(i)]  = {}
            phi[str(num_sub)][str(i)]  = {}
            mag[str(num_sub)][str(i)]  = {}
            chi_red[str(num_sub)][str(i)]  = {}
            mod_depth[str(num_sub)][str(i)] = {}
            surrogate[str(num_sub)][str(i)]  = {}
    
            
    
            for jf, freq in enumerate(int_band):  
                mod_depth[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['amp']
                surrogate[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['surrogate']
                phi[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['Fit'].best_values['phi']
                phi[str(num_sub)][str(i)][str(freq)] =  np.degrees(x[num_sub][str(i)][str(freq)][0]['Fit'].best_values['phi'])
                p[str(num_sub)][str(i)][str(freq)]  = x[num_sub][str(i)][str(freq)][0]['p']
                chi[str(num_sub)][str(i)][str(freq)]  = x[num_sub][str(i)][str(freq)][0]['Fit'].chisqr
                chi_red[str(num_sub)][str(i)][str(freq)]  =  x[num_sub][str(i)][str(freq)][0]['Fit'].redchi
                mag[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['Fit'].best_fit
                
    
    
    amp = {}
    phase = {}
    amp_erp_all = np.zeros([len(int_band), len(x)])
    phase_erp_all = np.zeros([len(int_band), len(x)])
    
    
    for i in range(len(labels)):
        amp[str(i)] = {}
        phase[str(i)] = {}
    
        
        for num_sub in range(len(x)):   
            amp_erp_all[:, num_sub]  = np.array(list(mod_depth[str(num_sub)][str(i)].values()))
            phase_erp_all[:, num_sub]  = np.array(list(phi[str(num_sub)][str(i)].values()))
        amp_array = np.mean(amp_erp_all, axis = 1)
        phase_array = np.mean(phase_erp_all, axis = 1)
        amp[str(i)] = amp_array
        phase[str(i)] = phase_array

    return(mod_depth, surrogate, phi)







def reading_cosine_function_parameters_just_phase(x, labels):
    
    p = {}
    chi = {}
    phi = {}
    mag = {}
    chi_red = {}
    mod_depth = {}
    surrogate = {}

    
    for num_sub in range(len(x)):
        p[str(num_sub)] = {}
        chi[str(num_sub)] = {}
        phi[str(num_sub)] = {}
        mag[str(num_sub)] = {}
        chi_red[str(num_sub)] = {}
        mod_depth[str(num_sub)] = {}
        surrogate[str(num_sub)] = {}
    
        
    
            
        for i in range(len(labels)): 
            p[str(num_sub)][str(i)]  = {}
            chi[str(num_sub)][str(i)]  = {}
            phi[str(num_sub)][str(i)]  = np.degrees(x[num_sub][str(i)][0]['Fit'].best_values['phi'])
            mag[str(num_sub)][str(i)]  = x[num_sub][str(i)][0]['Fit'].best_fit
            chi_red[str(num_sub)][str(i)] = x[num_sub][str(i)][0]['Fit'].redchi
            mod_depth[str(num_sub)][str(i)] = x[num_sub][str(i)][0]['amp']
            surrogate[str(num_sub)][str(i)]  = x[num_sub][str(i)][0]['surrogate']
    
     
    return(mod_depth, surrogate, phi)


















def subplot_torrecillos_2c_errorbar_ttest(mod_depth, surrogate, labels, save_folder): 
    plt.style.use('default')
    fig, ax = plt.subplots(1, len(labels), figsize=(20,4))
    for i_labels, name_labels in enumerate(labels):
        plt.style.use('default')
        int_band = range(len(np.arange(2, 18, 2)))
        int_band = np.arange(2, 18, 2)
        amp_erp_all = [] 
        surrogate_erp_all = []  
        for num_sub in range(len(mod_depth)):  
    
            amp_erp_all.append(np.array(list(mod_depth[str(num_sub)][str(i_labels)].values())))
            surrogate_erp_all.append(np.mean(np.array(list(surrogate[str(num_sub)][str(i_labels)].values())), axis =1))
            
        amp_erp_all_arr = np.array(amp_erp_all)
        surrogate_erp_all_arr = np.array(surrogate_erp_all)
        
        
    
        ax[i_labels].plot(int_band, np.mean(amp_erp_all_arr, axis = 0 ), color = 'k',  alpha=0.1)
        ax[i_labels].plot(int_band, np.mean(surrogate_erp_all_arr, axis = 0 ), color = 'r', alpha=0.1)
        e_mod= np.std(amp_erp_all_arr, axis = 0 )
        ax[i_labels].errorbar(int_band, np.mean(amp_erp_all_arr, axis = 0 ), e_mod, color = 'k',  linestyle='None', marker='o',   label = 'Real data')
        e_mod= np.std(surrogate_erp_all_arr, axis = 0 )
        ax[i_labels].errorbar(int_band, np.mean(surrogate_erp_all_arr, axis = 0 ), e_mod, color = 'r',  linestyle='None', marker='o',   label = 'Surrogate')
        ax[i_labels].set_title(f'{name_labels}, Individual Cosine Models', weight = 'bold')    
        ax[i_labels].set_ylabel('Modulation depth', weight = 'bold') 
        ax[i_labels].set_xlabel('Intensities (mA)', weight = 'bold')    
        ax[i_labels].legend(loc='upper right')
        ax[i_labels].set_ylim(bottom=0.2, top=1.5)
     
        
     
        
     
        
     

def subplot_torrecillos_2c_errorbar(mod_depth, surrogate, labels, save_folder): 
    plt.style.use('default')
    fig, ax = plt.subplots(1, len(labels), figsize=(20,4))
    for i_labels, name_labels in enumerate(labels):
        plt.style.use('default')
        int_band = np.arange(2, 18, 2)
           
        
        amp_erp_all = [] 
        surrogate_erp_all = []  
        for num_sub in range(len(mod_depth)):  
    
            amp_erp_all.append(np.array(list(mod_depth[str(num_sub)][str(i_labels)].values())))
            surrogate_erp_all.append(np.mean(np.array(list(surrogate[str(num_sub)][str(i_labels)].values())), axis =1))
            
        amp_erp_all_arr = np.array(amp_erp_all)
        surrogate_erp_all_arr = np.array(surrogate_erp_all)
        
        
    
        ax[i_labels].plot(int_band, np.mean(amp_erp_all_arr, axis = 0 ), color = 'k',  alpha=0.1)
        ax[i_labels].plot(int_band, np.mean(surrogate_erp_all_arr, axis = 0 ), color = 'r', alpha=0.1)
        e_mod= np.std(amp_erp_all_arr, axis = 0 )
        ax[i_labels].errorbar(int_band, np.mean(amp_erp_all_arr, axis = 0 ), e_mod, color = 'k',  linestyle='None', marker='o',   label = 'Real data')
        e_mod= np.std(surrogate_erp_all_arr, axis = 0 )
        ax[i_labels].errorbar(int_band, np.mean(surrogate_erp_all_arr, axis = 0 ), e_mod, color = 'r',  linestyle='None', marker='o',   label = 'Surrogate')
        ax[i_labels].set_title(f'{name_labels}, Individual Cosine Models', weight = 'bold')    
        ax[i_labels].set_ylabel('Modulation depth', weight = 'bold') 
        ax[i_labels].set_xlabel('Intensities (mA)', weight = 'bold')    
        ax[i_labels].legend(loc='upper right')
        ax[i_labels].set_ylim(bottom=0.0, top=1.7)
        
        
        
        threshold = 3
  
        # cluster permutation test
        T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([amp_erp_all_arr, surrogate_erp_all_arr], n_permutations=100, threshold=threshold, tail=1, n_jobs=1, out_type='mask')
    

    
        for i, i_int in enumerate(int_band):
            
            if (T_obs[i] > threshold and amp_erp_all_arr[i_labels, i] > surrogate_erp_all_arr[i_labels, i]): 
                ax[i_labels].plot(i_int,  1.3, '*', color = 'k')
                
    
    
    
    
# =============================================================================
#         for i_c, c in enumerate(clusters):
#             c = c[0]
#             if cluster_p_values[i_c] <= 0.05:
#                 ax[i_labels].plot(int_band[c.start],  1.3, '*', color = 'k')
#                 ax[i_labels].plot(int_band[c.stop - 1],  1.3, '*', color = 'k')
#                 ax[i_labels].text( int_band[c.start], 1.65, f'P-value = {cluster_p_values[i_c]}', color='k', weight = 'bold') 
#     
#         
#         
#     
# =============================================================================
    
    
    
    plt.show()  
        
        
    fig.tight_layout()    
    fig.savefig(save_folder+f'plot_torrecillos_2c_errorbar_subplot.svg')
    return fig





def phase_to_bin_class(x, phi, labels):
    
    bin_class_all = {}
    phi_tar_freq_all = {}
    
    for i in range(len(labels)):
        phi_tar_int = np.zeros([8, len(x)])
        for i_int, intensity in enumerate(np.arange(2,18,2)):
    
            for num_sub in range(len(x)):   
                phi_tar_int[i_int, num_sub]  = phi[str(num_sub)][str(i)][str(intensity)]
                if  phi[str(num_sub)][str(i)][str(intensity)] < 0:
                    phi_tar_int[i_int, num_sub]  = abs(phi[str(num_sub)][str(i)][str(intensity)]) 
                else:
                    phi_tar_int[i_int, num_sub]  =  360 - phi[str(num_sub)][str(i)][str(intensity)]
                    
    

        
    
    
    
    
    
    
    
        
        
        
        bin_num = 8
        bin_anticlockwise = np.linspace(0,360,int(bin_num+1))  # cover half of the circle -> with half of bin_num
        bin_clockwise = np.linspace(-360,0,int(bin_num+1)) 
        
        
        bin_class = np.nan*np.zeros(phi_tar_int.shape)
        phi_idx = np.nan*np.zeros(phi_tar_int.shape)
        
        for [row,col], phases in np.ndenumerate(phi_tar_int):
        # numbers correspond to the anti-clockwise unit circle eg. bin = 1 -> equals 22.5 deg phase for 16 bins
            if phases > 0:
                    idx = np.where(np.isclose(math.ceil(phi_tar_int[row,col]), bin_anticlockwise[:], atol=360/(bin_num*2)))
                    phi_idx[row,col] = bin_anticlockwise[idx]
                    # Returns a boolean array where two arrays are element-wise equal within a tolerance.
        # atol -> absolute tolerance level -> bin margins defined by 360° devided by twice the bin_num      
        # problem: rarely exactly between 2 bins -> insert nan
                    if len(idx) > 1:
                        idx = np.nan
                    bin_class[row,col] = idx[0]
        
            elif phases < 0:
                    idx, = np.where(np.isclose(math(phi_tar_int[row,col]), bin_clockwise[:], atol=360/(bin_num*2)))  
                    phi_idx[row,col] = bin_clockwise[idx]
                    if len(idx) > 1:
                        idx = np.nan      
                    bin_class[row,col] = idx[0]
                    
                    
                    
        
        # combine 360  and 0 together because they are basically the same
        bin_class[bin_class == 8] = 0
        bin_class_all[str(i)] = bin_class
        phi_tar_freq_all[str(i)] = phi_idx
    return(bin_class_all, phi_tar_freq_all)


def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return abs(x), np.angle(x)


def phase_optimal_per_sub(ax, phase_class, phase_val, phase_val_g,  title):
    plt.style.use('default')

    bin_num =8
    unique, counts = np.unique(phase_class, return_counts=True)
    phase_class_zero = np.zeros(8) 
    # adding zero
    for i_1, i_2 in enumerate(unique):
        phase_class_zero[int(i_2)] = counts[i_1]
    
    # number of equal bins
    bins = np.linspace(0.0, 2 * np.pi, bin_num + 1)
    
    #n, _, _ = plt.hist(bin_class[4, :], bins)
    # Why does plt.hist give me different values compared to when I do the bin class
    
    width = 2 * np.pi / bin_num
    #ax.plot(1, 1, 1, projection='polar')
    ax.bar(bins[:bin_num], phase_class_zero,  width=width, bottom=0.0)
    
    r, theta =R2P(np.sum(P2R(1,  np.radians(phase_val))))
    ax.plot([0,(theta)], [0, r ],  lw=3, color = 'red') 
    #ax.arrow(0,  0, theta, r , linewidth = 1.5,  head_width = 0.3, head_length = 1, fc='maroon', ec='maroon') 
    r_g, theta_g =R2P(np.sum(P2R(1,  np.radians(phase_val_g))))
    ax.plot([0,(theta_g)], [0, r_g ],  lw=2, color = 'k') 
    ax.set_ylim([0,8])
    ax.set_ylim([0,8])


    if np.degrees((theta))> 0:
        # 987 
        #ax.set_title(r"$\bf{" f'{title}' "}$" '\n'  f'{( np.degrees((theta)),2)}'   u'\N{DEGREE SIGN}'  '\n' f'{np.round( np.degrees((theta_g)),2)}' u'\N{DEGREE SIGN}' , color = {'red', 'k'}) 
       ax.text(1.75, 16,   f'{title}', weight = 'bold')
       ax.text(1.8,  14,   f'{np.round( np.degrees((theta)),2)}'u'\N{DEGREE SIGN}', color = 'maroon')
       ax.text(1.84, 12,   f'{np.round( phase_val_g,2)}'u'\N{DEGREE SIGN}')
    else:
         # To show positive degrees 360 + np.degrees((theta))
         
         #ax.set_title(r"$\bf{" f'{title}' "}$" '\n' f'{np.round(360 + np.degrees((theta)),2)}' u'\N{DEGREE SIGN}' '\n' f'{np.round(360 + np.degrees((theta_g)),2)}' u'\N{DEGREE SIGN}' ) 
         ax.text(1.75,16,  f'{title}', weight = 'bold')
         ax.text(1.8,  14,  f'{np.round(360 + np.degrees((theta)),2)}' u'\N{DEGREE SIGN}', color = 'maroon' )
         ax.text(1.84, 12,  f'{np.round(phase_val_g,2)}' u'\N{DEGREE SIGN}')
    #plt.tight_layout()
    plt.show()
    return(theta, theta_g)



def Boltzmann_sigmoid(x, Amplitude, Bias, Slope, Threshold):
    y = ( Amplitude * (Bias + ( (1-Bias) / (1 + np.exp(-Slope*(x-Threshold))))) )
    return y        

def GFP_comparison(intensity_range, a_GA, exdir, save_folder):
     ##################################################################
        fig, ax = plt.subplots()
        for i_intensity, intensity in enumerate(intensity_range):
            gfp = a_GA[str(intensity)].data.std(axis=0, ddof=0)
            if intensity == 2:
                colors = 'navy'; labels = '2 mA'
            elif intensity == 4:
                colors = 'dodgerblue'; labels = '4 mA'
            elif intensity == 6:
                colors = 'forestgreen'; labels = '6 mA'
            elif intensity == 8:
                colors = 'yellowgreen'; labels = '8 mA'
            elif intensity == 10:
                colors = 'yellow'; labels = '10 mA'
            elif intensity == 12:
                colors = 'orange'; labels = '12 mA'
            elif intensity == 14:
                colors = 'red'; labels = '14 mA'
            elif intensity == 16:
                colors = 'maroon'; labels = '16 mA'
                
                
                    
            ax.plot(a_GA[str(intensity)].times, gfp , color=colors, label = labels)
            ax.legend()
            ax.set_ylim(0, 2)
            ax.set(xlabel="Time (s)", ylabel="GFP (µV)")
            plt.show()
        if exdir[-6:-1] == 'Right':
            fig.suptitle('Right Stimulation Side')
            fig.savefig(save_folder+  'right/'+'GFP_comparison_R.svg')
        elif exdir[-5:-1] == 'Left':
            fig.suptitle('Left Stimulation Side')
            fig.savefig(save_folder+ 'left/'+'GFP_comparison_L.svg')
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
def GFP_comparison_L_low_med_high(intensity_range, a_GA, exdir, save_folder): 
    ##################################################################
    # GFP_comparison_L_low_med_high and mean of selected channels
    fig, ax = plt.subplots()    
    a_GA_low =   mne.grand_average([a_GA[str(2)], a_GA[str(4)]])  
    a_GA_med =   mne.grand_average([a_GA[str(6)], a_GA[str(8)], a_GA[str(10)]])   
    a_GA_high=   mne.grand_average([a_GA[str(12)], a_GA[str(14)], a_GA[str(16)]])  
    ax.plot(a_GA_low.times, a_GA_low.data.std(axis=0, ddof=0),  color = 'deepskyblue', label = 'low intensity (2, 4 mA)')
    ax.plot(a_GA_med.times, a_GA_med.data.std(axis=0, ddof=0),  color = 'darkorange', label = 'med intensity (6, 8, 10 mA)')
    ax.plot(a_GA_high.times, a_GA_high.data.std(axis=0, ddof=0), color = 'brown', label = 'high intensity (12, 14, 16 mA)')
    ax.set_ylim(0, 2)
    ax.legend()
    ax.set(xlabel="Time (s)", ylabel="GFP (µV)")
    plt.show()
    if exdir[-6:-1] == 'Right':
        fig.suptitle('Right Stimulation Side')
        fig.savefig(save_folder+  'right/'+'GFSelect_Epochs_intensityP_comparison_R_low_med_high.svg')
    elif exdir[-5:-1] == 'Left':
        fig.suptitle('Left Stimulation Side')
        fig.savefig(save_folder+ 'left/'+'GFP_comparison_L_low_med_high.svg')

    ##############################################
    if exdir[-5:-1] == 'Left': 
        fig, ax = plt.subplots()
        median_ch = {}
        for i_intensity, intensity in enumerate(intensity_range):
            median_ch[str(intensity)] = np.mean(a_GA[str(intensity)].pick_channels(['C4', 'CP4', 'C6', 'CP6']).data, axis  = 0)
            if intensity == 2:
                colors = 'navy'; labels = '2 mA'
            elif intensity == 4:
                colors = 'dodgerblue'; labels = '4 mA'
            elif intensity == 6:
                colors = 'forestgreen'; labels = '6 mA'
            elif intensity == 8:
                colors = 'yellowgreen'; labels = '8 mA'
            elif intensity == 10:
                colors = 'yellow'; labels = '10 mA'
            elif intensity == 12:
                colors = 'orange'; labels = '12 mA'
            elif intensity == 14:
                colors = 'red'; labels = '14 mA'
            elif intensity == 16:
                colors = 'maroon'; labels = '16 mA'
                
                
                    
            ax.plot(a_GA[str(intensity)].times, median_ch[str(intensity)] , color=colors, label = labels)
            ax.legend()
            ax.set_ylim(-2, 2)
            ax.set(xlabel="Time (s)", ylabel="Amplitude (µV)")
            plt.show()
        fig.suptitle('Left Stimulation Side(C4_C6_CP4_CP6)')
        fig.savefig(save_folder+ 'left/'+'C4_C6_CP4_CP6_comparison_L_low_med_high.svg')
        
        
# =============================================================================
#         fig, ax = plt.subplots()
#         ax.scatter(2, median_ch[str(2)][30], color = 'navy', label = '2 mA')
#         ax.scatter(4, median_ch[str(4)][30], color = 'dodgerblue', label = '4 mA')
#         ax.scatter(6, median_ch[str(6)][30], color = 'forestgreen', label = '6 mA')
#         ax.scatter(8, median_ch[str(8)][30], color = 'yellowgreen', label = '8 mA')
#         ax.scatter(10, median_ch[str(10)][30], color = 'yellow', label = '10 mA')
#         ax.scatter(12, median_ch[str(12)][30], color = 'orange', label = '12 mA')
#         ax.scatter(14, median_ch[str(14)][30], color = 'red', label = '14 mA')
#         ax.scatter(16, median_ch[str(16)][30], color = 'maroon', label = '16 mA')
#         ax.set_ylim(-2, 2)
#         ax.set_xlim(1, 17)
#         ax.legend(ncol = 4, loc="lower right")
#         plt.show()
#         fig.suptitle('Left Stimulation Side(C4_C6_CP4_CP6)')
#         fig.savefig(save_folder+ 'left/'+'_scatter_C3_C5_CP3_CP5_comparison_L_low_med_high.svg')
# =============================================================================
        
        
    if exdir[-6:-1] == 'Right': 
        fig, ax = plt.subplots()
        median_ch = {}
        for i_intensity, intensity in enumerate(intensity_range):
            median_ch[str(intensity)] = np.mean(a_GA[str(intensity)].pick_channels(['C3', 'CP3', 'CP3', 'CP5']).data, axis = 0)
            if intensity == 2:
                colors = 'navy'; labels = '2 mA'
            elif intensity == 4:
                colors = 'dodgerblue'; labels = '4 mA'
            elif intensity == 6:
                colors = 'forestgreen'; labels = '6 mA'
            elif intensity == 8:
                colors = 'yellowgreen'; labels = '8 mA'
            elif intensity == 10:
                colors = 'yellow'; labels = '10 mA'
            elif intensity == 12:
                colors = 'orange'; labels = '12 mA'
            elif intensity == 14:
                colors = 'red'; labels = '14 mA'
            elif intensity == 16:
                colors = 'maroon'; labels = '16 mA'
                
                
                    
            ax.plot(a_GA[str(intensity)].times, median_ch[str(intensity)] , color=colors, label = labels)
            ax.legend()
            #ax.set_ylim(-2, 3)
            ax.set(xlabel="Time (s)", ylabel="Amplitude (µV)")
            plt.show()
        fig.suptitle('Right Stimulation Side(C3_C5_CP3_CP5)')
        fig.savefig(save_folder+ 'right/'+'C3_C5_CP3_CP5_comparison_L_low_med_high.svg')
        
    return(median_ch)




def Boltzmann_sigmoid_SEP(median_ch, exdir, save_folder):
        
        tp = 30 # interested time point
        y = np.array([median_ch[str(2)][tp], median_ch[str(4)][tp], median_ch[str(6)][tp], median_ch[str(8)][tp],\
                      median_ch[str(10)][tp], median_ch[str(12)][tp], median_ch[str(14)][tp], median_ch[str(16)][tp]])

        m1 = lmfit.models.Model(Boltzmann_sigmoid)
        parms = m1.make_params()
        parms['Amplitude'].set(1., min=0., max=10.)
        parms['Bias'].set(0.001, min=0.001, max=.01)
        parms['Slope'].set(.9, min=0, max=10)
        parms['Threshold'].set(6., min=1., max=16.)    
        x= np.arange(2, 18, 2)
        
        
        out = m1.fit(y, parms, x = x, method = 'leastsq', nan_policy = 'omit')
        
        fit_curve = Boltzmann_sigmoid(x, Amplitude=out.best_values['Amplitude'], Bias = out.best_values['Bias'], 
                                      Slope = out.best_values['Slope'], Threshold = out.best_values['Threshold'])
        
        
        fig, ax = plt.subplots() 
        ax.scatter(2, median_ch[str(2)][tp], color = 'navy', label = '2 mA')
        ax.scatter(4, median_ch[str(4)][tp], color = 'dodgerblue', label = '4 mA')
        ax.scatter(6, median_ch[str(6)][tp], color = 'forestgreen', label = '6 mA')
        ax.scatter(8, median_ch[str(8)][tp], color = 'yellowgreen', label = '8 mA')
        ax.scatter(10, median_ch[str(10)][tp], color = 'yellow', label = '10 mA')
        ax.scatter(12, median_ch[str(12)][tp], color = 'orange', label = '12 mA')
        ax.scatter(14, median_ch[str(14)][tp], color = 'red', label = '14 mA')
        ax.scatter(16, median_ch[str(16)][tp], color = 'maroon', label = '16 mA')
        #ax.set_ylim(-1, 2)
        ax.set_xlim(1, 17)
        ax.legend(ncol = 4, loc="lower right")
        ax.set_xlabel('Intensities (mA)')
        ax.set_ylabel('Amplitude  (µV)')
        ax.plot(x, fit_curve)
        plt.show()
        
        fig.suptitle('Greatest slope at {:.1f} (mA) Intensity'.format(x[np.argmax(np.diff(fit_curve))]+2))
        if exdir[-6:-1] == 'Right':
            fig.savefig(save_folder+ 'right/'+'curve_fitting_C3_C5_CP3_CP5.svg')
        elif exdir[-5:-1] == 'Left':
            fig.savefig(save_folder+ 'left/'+'curve_fitting_C4_C6_CP4_CP6.svg')
            
            
            
            
            
        
            
def Boltzmann_sigmoid_SEP_emg(a_GA, exdir, save_folder, name):
        intensity_range = np.arange(2, 18, 2)
        EMG = {}
        EMG_p_p = {}
        for i_intensity, intensity in enumerate(intensity_range):
            EMG[str(intensity)]= a_GA[str(intensity)].data - (np.mean(a_GA[str(intensity)].data))

            mn=np.min(EMG[str(intensity)][:,900:1100])
            mx=np.max(EMG[str(intensity)][:,900:1100])
            EMG_p_p[str(intensity)] = mx - mn
            if np.isnan(EMG_p_p[str(intensity)]) == True:
                EMG_p_p[str(intensity)] = 0
                

        #y = np.log(np.array(list(EMG_p_p.values())))
        y = np.array(list(EMG_p_p.values()))
        

        m1 = lmfit.models.Model(Boltzmann_sigmoid)
        parms = m1.make_params()
        parms['Amplitude'].set(0., min=1., max=1000.)
        parms['Bias'].set(0.001, min=0.1, max=10)
        parms['Slope'].set(.9, min=0, max=10)
        parms['Threshold'].set(6., min=1., max=16.)
        x= np.arange(2, 18, 2)
        
        
        out = m1.fit(y, parms, x = x, method = 'leastsq', nan_policy = 'omit')
        
        fit_curve = Boltzmann_sigmoid(x, Amplitude=out.best_values['Amplitude'], Bias = out.best_values['Bias'], 
                                      Slope = out.best_values['Slope'], Threshold = out.best_values['Threshold'])
        
        EMG_p_p = np.log(list(EMG_p_p.values()))
        fig, ax = plt.subplots() 
        ax.scatter(2, y[0],  color = 'navy', label = '2 mA')
        ax.scatter(4, y[1],  color = 'dodgerblue', label = '4 mA')
        ax.scatter(6, y[2],  color = 'forestgreen', label = '6 mA')
        ax.scatter(8, y[3],  color = 'yellowgreen', label = '8 mA')
        ax.scatter(10,y[4], color = 'yellow', label = '10 mA')
        ax.scatter(12,y[5], color = 'orange', label = '12 mA')
        ax.scatter(14,y[6], color = 'red', label = '14 mA')
        ax.scatter(16,y[7], color = 'maroon', label = '16 mA')
        #ax.set_ylim(-1, 2)
        ax.set_xlim(1, 17)
        ax.legend(ncol = 4, loc="lower right")
        ax.set_xlabel('Intensities (mA)')
        ax.set_ylabel('Amplitude  (µV)')
        ax.plot(x, fit_curve)
        plt.show()
        
        fig.suptitle('Greatest slope at {:.1f} (mA) Intensity'.format(x[np.argmax(np.diff(fit_curve))]+2))
        if exdir[-6:-1] == 'Right':
            fig.savefig(save_folder + 'right/' + f'{name[0:4]}'   +'_EDC_R.svg')
        elif exdir[-5:-1] == 'Left':
            fig.savefig(save_folder+ 'left/'+  f'{name[0:4]}'  +'_EDC_L.svg')
            
            
            
            
            
            
def EMG_time_course(a_GA, exdir, save_folder, name):  
    intensity_range = np.arange(2, 18, 2)
    fig, ax = plt.subplots()

    for i_intensity, intensity in enumerate(intensity_range):
        if intensity == 2:
            colors = 'navy'; labels = '2 mA'
        elif intensity == 4:
            colors = 'dodgerblue'; labels = '4 mA'
        elif intensity == 6:
            colors = 'forestgreen'; labels = '6 mA'
        elif intensity == 8:
            colors = 'yellowgreen'; labels = '8 mA'
        elif intensity == 10:
            colors = 'yellow'; labels = '10 mA'
        elif intensity == 12:
            colors = 'orange'; labels = '12 mA'
        elif intensity == 14:
            colors = 'red'; labels = '14 mA'
        elif intensity == 16:
            colors = 'maroon'; labels = '16 mA'
            
            
        
        ax.plot(a_GA[str(intensity)].times, a_GA[str(intensity)]._data.T , color=colors, label = labels)
    ax.legend(ncol = 2, loc="upper right")
    ax.set(xlabel="Time (s)", ylabel="Amplitude (µV)")
    plt.show()
    
    if exdir[-6:-1] == 'Right':
        fig.suptitle(f'{ name[0:4]} _Right')
        fig.savefig(save_folder + 'right/' + name[0:4]+ '_time_course' +'.svg')
    elif exdir[-5:-1] == 'Left':
        fig.suptitle(f'{ name[0:4]} _Left')
        fig.savefig(save_folder + 'left/' + name[0:4]+ '_time_course' +'.svg')
        
        
        
        
def p30_amp(exdir_epoch_both, intensity_range, a_GA):
    p30 = {}
    for i_exdir, exdir in enumerate(exdir_epoch_both):
        p30[str(i_exdir)] = {}
        if i_exdir == 0:
            for i_intensity, intensity in enumerate(intensity_range):
                p30[str(i_exdir)][str(i_intensity)] = np.mean(np.mean((a_GA[str(i_exdir)][str(intensity)].pick_channels(['C1', 'CP1'])._data[:, 28:33]), axis =0))
        elif i_exdir == 1:
            for i_intensity, intensity in enumerate(intensity_range):
                p30[str(i_exdir)][str(i_intensity)] = np.mean(np.mean((a_GA[str(i_exdir)][str(intensity)].pick_channels(['C2', 'CP2'])._data[:, 28:33]), axis =0))
    
    return(p30)




def p50_amp(exdir_epoch_both, intensity_range, a_GA):
    p50 = {}
    #N1 = {}
    for i_exdir, exdir in enumerate(exdir_epoch_both):
    
        p50[str(i_exdir)] = {}
        #N1[str(i_exdir)] = {}
        if i_exdir == 0:
            for i_intensity, intensity in enumerate(intensity_range):
                p50[str(i_exdir)][str(i_intensity)] = np.mean(np.mean((a_GA[str(i_exdir)][str(intensity)].pick_channels(['C3', 'C5', 'CP3', 'CP5'])._data[:, 47:52]), axis =0))
                #N1[str(i_exdir)][str(i_intensity)] = np.mean(np.mean((a_GA[str(i_exdir)][str(intensity)].pick_channels(['F1', 'F2', 'Fz', 'FC1', 'FC2'])._data[:, 47:52]), axis =0))
        elif i_exdir == 1:
            for i_intensity, intensity in enumerate(intensity_range):
                p50[str(i_exdir)][str(i_intensity)] = np.mean(np.mean((a_GA[str(i_exdir)][str(intensity)].pick_channels(['C4', 'C6', 'CP4', 'CP6'])._data[:, 47:52]), axis =0))
                #N1[str(i_exdir)][str(i_intensity)] = np.mean(np.mean((a_GA[str(i_exdir)][str(intensity)].pick_channels(['F1', 'F2', 'Fz', 'FC1', 'FC2'])._data[:, 47:52]), axis =0))

    return(p50)




def p30_amp_ind(exdir_epoch_both, intensity_range, a_dict):
    n_sub_l =  np.arange(0, 8, 1) 
    n_sub_r =  np.arange(0, 11, 1) 
    p30 = {}
    for i_exdir, exdir in enumerate(exdir_epoch_both):
        if i_exdir == 0:
            p30[str(i_exdir)] = np.zeros([ len(intensity_range), len(n_sub_l)])   
            for i_intensity, intensity in enumerate(intensity_range):
                 for i_sub, _ in enumerate(n_sub_l):
                
                     p30[str(i_exdir)][i_intensity, i_sub] =np.mean(a_dict[str(i_exdir)][str(intensity)][i_sub].pick_channels(['C1', 'CP1'])._data[:, 28:33])
       
    
        elif i_exdir == 1:
            p30[str(i_exdir)] = np.zeros([ len(intensity_range), len(n_sub_r)])   
            for i_intensity, intensity in enumerate(intensity_range):
                for i_sub, _ in enumerate(n_sub_r):
                    p30[str(i_exdir)][i_intensity, i_sub] = np.mean(a_dict[str(i_exdir)][str(intensity)][i_sub].pick_channels(['C2', 'CP2'])._data[:, 28:33])
    return p30
    
    
    
def p50_amp_ind(exdir_epoch_both, intensity_range, a):
    n_sub_l =  np.arange(0, 8, 1) 
    n_sub_r =  np.arange(0, 11, 1) 
    p50 = {}
    for i_exdir, exdir in enumerate(exdir_epoch_both):
        if i_exdir == 0:
            p50[str(i_exdir)] = np.zeros([ len(intensity_range), len(n_sub_l)])   
            for i_intensity, intensity in enumerate(intensity_range):
                 for i_sub, _ in enumerate(n_sub_l):
                     p50[str(i_exdir)][i_intensity, i_sub] =np.mean(a[str(i_exdir)][str(intensity)][i_sub].pick_channels(['C4', 'C6', 'CP4', 'CP6'])._data[:, 47:52])
       
    
        elif i_exdir == 1:
            p50[str(i_exdir)] = np.zeros([ len(intensity_range), len(n_sub_r)])   
            for i_intensity, intensity in enumerate(intensity_range):
                for i_sub, _ in enumerate(n_sub_r):
                    p50[str(i_exdir)][i_intensity, i_sub] = np.mean(a[str(i_exdir)][str(intensity)][i_sub].pick_channels(['C3', 'C5', 'CP3', 'CP5'])._data[:, 47:52])
    return p50
    
def LME_mean_intensity(sub_names_L, sub_names_R, peaks_l, peaks_r, i_l, st_ind_l, st_ind_r):

    all_res = []    
    dfl =pd.DataFrame(np.zeros((1, 4)), columns=['sub', 'data', 'visit', 'stim_side'])
    
    for _, side in enumerate(['L', 'R']):
        for _,time_point in enumerate(['v2', 'v3', 'v4', 'v5', 'v6']):
            i=i_l
            if side == 'L':
                res = pd.DataFrame((sub_names_L[str(time_point)], np.mean(peaks_l[str(time_point)][:, st_ind_l[str(i)], i], axis =1), list(np.repeat(time_point, len(sub_names_L[str(time_point)]))), list(np.repeat('L', len(sub_names_L[str(time_point)])))    )).T
                all_res.append(res)
            elif side == 'R': 
                 res = pd.DataFrame((sub_names_R[str(time_point)], np.mean(peaks_r[str(time_point)][:, st_ind_r[str(i)], i], axis =1), list(np.repeat(time_point, len(sub_names_R[str(time_point)]))), list(np.repeat('R', len(sub_names_R[str(time_point)])))    )).T
                 all_res.append(res)
            df_res = pd.concat(all_res)    
    
    df_res.rename(columns={0: 'sub', 1:'data', 2:'visit', 3:'stim_side'})  
    
    if i == 0:    
        df_res.to_excel("P1.xlsx", sheet_name='P1')     
    elif i ==1:
        df_res.to_excel("N1.xlsx", sheet_name='N1')
    elif i ==2:
        df_res.to_excel("P2.xlsx", sheet_name='P2')
    
def LME_mean_intensity_FuMe(sub_names_L, sub_names_R, peaks_l, peaks_r, i_l, st_ind_l, st_ind_r):

    all_res = []    
    dfl =pd.DataFrame(np.zeros((1, 4)), columns=['sub', 'data', 'visit', 'stim_side'])
    
    for _, side in enumerate(['L', 'R']):
        for _,time_point in enumerate(['v2', 'v3', 'v4', 'v5', 'v6']):
            i=i_l
            if side == 'L':
                res = pd.DataFrame((sub_names_L[str(time_point)], np.mean(peaks_l[str(time_point)][:, st_ind_l[str(i)], i], axis =1), list(np.repeat(time_point, len(sub_names_L[str(time_point)]))), list(np.repeat('L', len(sub_names_L[str(time_point)])))    )).T
                all_res.append(res)
            elif side == 'R': 
                 res = pd.DataFrame((sub_names_R[str(time_point)], np.mean(peaks_r[str(time_point)][:, st_ind_r[str(i)], i], axis =1), list(np.repeat(time_point, len(sub_names_R[str(time_point)]))), list(np.repeat('R', len(sub_names_R[str(time_point)])))    )).T
                 all_res.append(res)
            df_res = pd.concat(all_res)    
    
    df_res.rename(columns={0: 'sub', 1:'data', 2:'visit', 3:'stim_side'})  
    
    if i == 0:    
        df_res.to_excel("P1.xlsx", sheet_name='P1')     
    elif i ==1:
        df_res.to_excel("N1.xlsx", sheet_name='N1')
    elif i ==2:
        df_res.to_excel("P2.xlsx", sheet_name='P2')   
    
def LME_intensity(sub_names_L, sub_names_R, peaks_l, peaks_r, i_l, st_ind_l, st_ind_r):   
    all_res = []    
    dfl =pd.DataFrame(np.zeros((1, 5)), columns=['sub', 'data', 'visit', 'stim_side', 'stim_intensity'])
    
    for _, side in enumerate(['L', 'R']):
        for _,time_point in enumerate(['v2', 'v3', 'v4', 'v5', 'v6']):
            i=i_l # i label, counter of the peaks 
            if side == 'L':
                for i_int, intensity in enumerate(np.arange(2, 18, 2)): 
                    #print(i_int)
                    #res = pd.DataFrame((np.tile(sub_names_L[str(time_point)], len(np.arange(2, 18, 2))), np.mean(peaks_l[str(time_point)][:, st_ind_l[str(i)], i], axis =1)[ :, i_int], list(np.repeat(time_point, len(sub_names_L[str(time_point)])* len(np.arange(2, 18, 2)) )), list(np.repeat('L', len(sub_names_L[str(time_point)])))    )).T
                    res = pd.DataFrame(((sub_names_L[str(time_point)], len(np.arange(2, 18, 2)))[0], np.mean(peaks_l[str(time_point)][:, st_ind_l[str(i)], i], axis =1)[ :, i_int], list(np.repeat(time_point, len(sub_names_L[str(time_point)]))),   list(np.repeat('L', len(sub_names_L[str(time_point)])))  , list(np.repeat(intensity, len(sub_names_L[str(time_point)])))   )).T
    
                    all_res.append(res)
            elif side == 'R': 
                for i_int, intensity in enumerate(np.arange(2, 18, 2)): 
                 #print(i_int)
                     res = pd.DataFrame(((sub_names_R[str(time_point)], len(np.arange(2, 18, 2)))[0], np.mean(peaks_r[str(time_point)][:, st_ind_r[str(i)], i], axis =1)[ :, i_int], list(np.repeat(time_point, len(sub_names_R[str(time_point)]))),  list(np.repeat('R', len(sub_names_R[str(time_point)]))),list(np.repeat(intensity, len(sub_names_R[str(time_point)])))   )).T
        
                     all_res.append(res)
                 
                 
                 
    df_res = pd.concat(all_res)  
    df_res = df_res.dropna()   
    
    df_res.rename(columns={0: 'sub', 1:'data', 2:'visit', 3:'stim_side', 4: 'intensity'},  inplace=True)  


    
    if i == 0:    
        df_res.to_excel("P1_intensity.xlsx", sheet_name='P1')     
    elif i ==1:
        df_res.to_excel("N1_intensity.xlsx", sheet_name='N1')
    elif i ==2:
        df_res.to_excel("P2_intensity.xlsx", sheet_name='P2')    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

def HC_fitting_cosine_plotting_left_right_ind(win_erps, labels, com_ind_l, com_ind_r, phi_array_deg_correct_l, phi_array_deg_correct_r, exdir_epoch_l, exdir_epoch_r, save_folder):
    ####################################################################################
    # Left
    ind_amplitudes_cosine_l, ind_cosine_fit_l ,ind_subjects_names_l = individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
    mod_depth_l, surrogate, phi = reading_cosine_function_parameters(ind_cosine_fit_l, labels) 
    subplot_torrecillos_2c_errorbar(mod_depth_l, surrogate, labels, save_folder + 'left/')
    bin_class_all, phi_tar_freq_all =  phase_to_bin_class(ind_cosine_fit_l, phi, labels)
        
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
             theta[str(intensity)], theta_g[str(intensity)] = phase_optimal_per_sub(ax,  bin_class_all[str(row)][intensity,:],  phi_tar_freq_all[str(row)][intensity,:], phi_array_deg_correct_l[intensity,row], titles[intensity])   
     
    fig.savefig(save_folder  + 'left/' +'optimal_phase_distribution.svg')         
       
    
    
    ####################################################################################
    # Right
    ind_amplitudes_cosine_r, ind_cosine_fit_r ,ind_subjects_names_r = individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_r)
    mod_depth_r, surrogate, phi = reading_cosine_function_parameters(ind_cosine_fit_r, labels) 
    subplot_torrecillos_2c_errorbar(mod_depth_r, surrogate, labels, save_folder + 'right/')
    
    
    
    
    bin_class_all, phi_tar_freq_all =  phase_to_bin_class(ind_cosine_fit_r, phi, labels)
        
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
             theta[str(intensity)], theta_g[str(intensity)] = phase_optimal_per_sub(ax,  bin_class_all[str(row)][intensity,:],  phi_tar_freq_all[str(row)][intensity,:], phi_array_deg_correct_r[intensity,row], titles[intensity])   
     
    fig.savefig(save_folder  + 'right/' +'optimal_phase_distribution.svg')         
    return mod_depth_l, mod_depth_r, ind_cosine_fit_l, ind_cosine_fit_r
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def HC_fitting_cosine_plotting_left_right_group(win_erps, labels, com_ind_l, com_ind_r, exdir_epoch_l, exdir_epoch_r, save_folder):
    # Cosine fitting for the right hemisphere for each intensity stim_int = np.arange(2, 18, 2)
    cosinefit_r, amplitudes_cosine_r, pvalues_cosine_r = cosine_fit_intensity_phase_overlapping_chs( win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)
    
    
    # Further analysis just for the right hemisphere
    
    # Circular correlation
    phi_array_deg_r = Circ_corr(cosinefit_r, labels)
    # Modulation depth bar plot
    p_val_r, phi_array_deg_correct_r, amp_df_r = amp_p_chi(cosinefit_r, labels, phi_array_deg_r, save_folder + 'com_ch/right/')
    # Plotting best cosine fit and data
    title = 'Group'
    best_fit_plot(cosinefit_r, labels, phi_array_deg_r, save_folder + 'com_ch/right/' , title, [-1,1])
    
    fig_2c_scatter_sub_plot(cosinefit_r, labels, p_val_r, save_folder + 'com_ch/right/')
    
    
    
    
    # Cosine fitting for the left hemisphere for each intensity stim_int = np.arange(2, 18, 2)
    cosinefit_l, amplitudes_cosine_l, pvalues_cosine_l = cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
    # Further analysis just for the right hemisphere
    
    # Circular correlation
    phi_array_deg_l = Circ_corr(cosinefit_l, labels)
    # Modulation depth bar plot
    p_val_l, phi_array_deg_correct_l, amp_df_l = amp_p_chi(cosinefit_l, labels, phi_array_deg_l, save_folder + 'com_ch/left/')
    # Plotting best cosine fit and data
    title = 'Group'
    best_fit_plot(cosinefit_l, labels, phi_array_deg_l, save_folder + 'com_ch/left/', title, [-1,1])
    fig_2c_scatter_sub_plot(cosinefit_l, labels, p_val_l, save_folder + 'com_ch/left/')
    
    
    
    
    cosinefit_r_avg_int, amplitudes_cosine_r_avg_int, pvalues_cosine_r_avg_int, surrogate_r_avg  = cosine_fit_phase_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)
    cosinefit_l_avg_int, amplitudes_cosine_l_avg_int, pvalues_cosine_l_avg_int, surrogate_l_avg  = cosine_fit_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
    
    
    
    bar_plot_avg_intensity(labels, amplitudes_cosine_r_avg_int, surrogate_r_avg, pvalues_cosine_r_avg_int, 'Group Cosine Model for avg Intensities' , save_folder + 'com_ch/right/')
    bar_plot_avg_intensity(labels, amplitudes_cosine_l_avg_int, surrogate_l_avg, pvalues_cosine_l_avg_int, 'Group Cosine Model for avg Intensities' , save_folder + 'com_ch/left/')


    return phi_array_deg_correct_l, phi_array_deg_correct_r, amp_df_l, amp_df_r, p_val_r, p_val_l
    
    
    
    

def com_chs_and_plotting_clusters(labels, ch_names_l, ch_names_r, ch_names_l_hc, ch_names_r_hc, t_l, t_r, pvals_all_l, pvals_all_r, pos, save_folder):

    ch_names = channel_names()
    com_ch_l = {}
    for i_ch, chs in enumerate(ch_names_l_hc):
        com_ch_l[str(i_ch)], _, _ = np.intersect1d(chs ,ch_names_l[str('v2')][i_ch], return_indices=True)
    
    
    
    com_ch_r = {}
    for i_ch, chs in enumerate(ch_names_r_hc):
        com_ch_r[str(i_ch)], _, _ = np.intersect1d(chs ,ch_names_r[str('v2')][i_ch], return_indices=True)
    
    

    
    com_ind_l = {}
    com_ind_r = {}
    mask_com_l = np.zeros((64,3))
    mask_com_r = np.zeros((64,3))
    for i, _ in enumerate(labels):
        _, com_ind_l[str(i)], _ = np.intersect1d(ch_names ,com_ch_l[str(i)], return_indices=True)
        _, com_ind_r[str(i)], _ = np.intersect1d(ch_names ,com_ch_r[str(i)], return_indices=True)
      
        np.put(mask_com_l[:,i], com_ind_l[str(i)], np.ones(len(com_ind_l[str(i)])))    
        np.put(mask_com_r[:,i], com_ind_r[str(i)], np.ones(len(com_ind_r[str(i)])))   
        
    t_r[str('v2')][t_r[str('v2')] > 10] = 10
    t_r[str('v2')][t_r[str('v2')] < -10] = -10    
    fig_com_r =  plot_topomap_peaks_second_v_com(t_r[str('v2')], mask_com_r, ch_names, pvals_all_r[str('v2')], [-11,11], pos, i_intensity = 'all')
    fig_com_l = plot_topomap_peaks_second_v_com((t_l[str('v2')]), mask_com_l, ch_names, pvals_all_l[str('v2')], [-11,11], pos, i_intensity = 'all')
    fig_com_l.savefig(save_folder + '/left/'+'l_com_st_hc.svg')
    fig_com_r.savefig(save_folder + '/right/'+'r_com_st_hc.svg')
    
    return com_ind_l, com_ind_r    








def ST_fitting_cosine_plotting_left_right_group(win_erps, labels, com_ind_l, com_ind_r, exdir_epoch_l, exdir_epoch_r, save_folder):

    # Cosine fitting for the right hemisphere for each intensity stim_int = np.arange(2, 18, 2)
    cosinefit_r, amplitudes_cosine_r, pvalues_cosine_r = cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)
    
    
    # Further analysis just for the right hemisphere
    
    # Circular correlation
    phi_array_deg_r = Circ_corr(cosinefit_r, labels)
    # Modulation depth bar plot
    p_val_r, phi_array_deg_correct_r_st, amp_df_r = amp_p_chi(cosinefit_r, labels, phi_array_deg_r, save_folder + '/right/')
    # Plotting best cosine fit and data
    title = 'Group'
    best_fit_plot(cosinefit_r, labels, phi_array_deg_r, save_folder + '/right/' , title, [-1,1])
    
    fig_2c_scatter_sub_plot(cosinefit_r, labels, p_val_r, save_folder + '/right/')
    
    
    # Cosine fitting for the left hemisphere for each intensity stim_int = np.arange(2, 18, 2)
    cosinefit_l, amplitudes_cosine_l, pvalues_cosine_l = cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
    # Further analysis just for the right hemisphere
    
    # Circular correlation
    phi_array_deg = Circ_corr(cosinefit_l, labels)
    # Modulation depth bar plot
    p_val_l, phi_array_deg_correct_l_st, amp_df_l = amp_p_chi(cosinefit_l, labels, phi_array_deg, save_folder + '/left/')
    # Plotting best cosine fit and data
    title = 'Group'
    best_fit_plot(cosinefit_l, labels, phi_array_deg, save_folder + 'left/', title, [-1,1])
    
    fig_2c_scatter_sub_plot(cosinefit_l, labels, p_val_l, save_folder + 'left/')
    
    cosinefit_r_avg_int, amplitudes_cosine_r_avg_int, pvalues_cosine_r_avg_int, surrogate_r_avg  = cosine_fit_phase_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)
    cosinefit_l_avg_int, amplitudes_cosine_l_avg_int, pvalues_cosine_l_avg_int, surrogate_l_avg  = cosine_fit_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
    
    
    
    bar_plot_avg_intensity(labels, amplitudes_cosine_r_avg_int, surrogate_r_avg, pvalues_cosine_r_avg_int, 'Group Cosine Model for avg Intensities' , save_folder + 'right/')
    bar_plot_avg_intensity(labels, amplitudes_cosine_l_avg_int, surrogate_l_avg, pvalues_cosine_l_avg_int, 'Group Cosine Model for avg Intensities' , save_folder + 'left/')
    
    return phi_array_deg_correct_l_st, phi_array_deg_correct_r_st,  amp_df_l, amp_df_r, pvalues_cosine_r, pvalues_cosine_l















def ST_fitting_cosine_plotting_left_right_ind(win_erps, labels, com_ind_l, com_ind_r, phi_array_deg_correct_l, phi_array_deg_correct_r, exdir_epoch_l, exdir_epoch_r, save_folder):
    
    
        
        
    ind_amplitudes_cosine_l, ind_cosine_fit_l ,ind_subjects_names_l = individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_l.values()), exdir_epoch_l)
    
    

    mod_depth, surrogate, phi = reading_cosine_function_parameters(ind_cosine_fit_l, labels) 
    subplot_torrecillos_2c_errorbar(mod_depth, surrogate, labels, save_folder + 'left/')
    bin_class_all, phi_tar_freq_all =  phase_to_bin_class(ind_cosine_fit_l, phi, labels)
        
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
             theta[str(intensity)], theta_g[str(intensity)] = phase_optimal_per_sub(ax,  bin_class_all[str(row)][intensity,:],  phi_tar_freq_all[str(row)][intensity,:], phi_array_deg_correct_l[intensity,row], titles[intensity])   
     
        
     
        
     
    fig.savefig(save_folder  + 'left/' +'optimal_phase_distribution.svg')    
    
    
    ###############################################################################
    # Right
    
    
    
    
    
    
    ind_amplitudes_cosine_r, ind_cosine_fit_r ,ind_subjects_names_r = individual_cosine_fit_intensity_phase_overlapping_chs(win_erps, labels, list(com_ind_r.values()), exdir_epoch_r)

    
    mod_depth, surrogate, phi = reading_cosine_function_parameters(ind_cosine_fit_r, labels) 
    subplot_torrecillos_2c_errorbar(mod_depth, surrogate, labels, save_folder + 'right/')
    bin_class_all, phi_tar_freq_all = phase_to_bin_class(ind_cosine_fit_r, phi, labels)
        
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
             theta[str(intensity)], theta_g[str(intensity)] = phase_optimal_per_sub(ax,  bin_class_all[str(row)][intensity,:],  phi_tar_freq_all[str(row)][intensity,:], phi_array_deg_correct_r[intensity,row], titles[intensity])   
     
    fig.savefig(save_folder  + 'right/' +'optimal_phase_distribution.svg')   
    
    
    
    return ind_amplitudes_cosine_l, ind_amplitudes_cosine_r, ind_cosine_fit_l, ind_cosine_fit_r
    


def hc_labels_intensity(labels, st_ind, peaks_hc, name):
    peaks_hc_mean_phase =  np.mean(peaks_hc, (-2))
    df = {}
    for i_l, label in enumerate(labels):
        all_res = []    
        i = i_l
        for i_int, intensity in enumerate(np.arange(2, 18, 2)):    
            res = pd.DataFrame(np.mean(peaks_hc_mean_phase[:, st_ind[str(i_l)],i_l, i_int ], axis = 1), list(np.repeat(intensity, len(np.mean(peaks_hc_mean_phase[:, st_ind[str(i_l)],i_l, i_int ], axis = 1)))))
            all_res.append(res)
            df_res = pd.concat(all_res)  
            df_res = df_res.dropna()  
        if i == 0:
            df_res.rename(columns={0: 'P1'}, inplace=True)
        elif i == 1:
            df_res.rename(columns={0: 'N1'}, inplace=True)
        elif i== 2:
            df_res.rename(columns={0: 'P2'}, inplace=True)
        df[str(i)] = df_res 
    df_all = pd.concat([df[str(0)], df[str(1)], df[str(2)]],  axis = 1)       
     
    df_all.to_excel(f'{name}')  
    return(pd.concat([df[str(0)], df[str(1)], df[str(2)]],  axis = 1))






def peaks_plot(peaks, com_ind, labels, title, save_folder):
    from sklearn import preprocessing as pre
    peaks_hc_mean_phase =  np.mean(peaks, (-2))
    df = {}
    for i_l, label in enumerate(labels):
        df[str(i_l)] = {}
        for i_int, intensity in enumerate(np.arange(2, 18, 2)):    
             #df[str(i_l)][str(i_int)] = np.mean(stats.zscore(peaks_hc_mean_phase[:, com_ind[str(i_l)],i_l, i_int ], axis = 1), axis = 1)
             df[str(i_l)][str(i_int)] = stats.zscore(np.mean(peaks_hc_mean_phase[:, com_ind[str(i_l)],i_l, i_int ], axis = 1), axis = 0)
             df[str(i_l)][str(i_int)] = pre.MinMaxScaler().fit_transform(np.mean(peaks_hc_mean_phase[:, com_ind[str(i_l)],i_l, i_int ], axis = 1).reshape(-1, 1))
             #df[str(i_l)][str(i_int)] = np.mean(preprocessing.normalize(peaks_hc_mean_phase[:, com_ind[str(i_l)],i_l, i_int ], axis = 1), axis = 1)
    fig = plt.figure()
    
    df_mean = np.zeros([len((np.arange(2, 18, 2))), len(labels)])
    for i_labels, name_labels in enumerate(labels):
        if  i_labels == 0 :
            color = 'maroon'; component = 'P1'
        elif i_labels == 1:
            color = 'navy'; component = 'N1'
        elif i_labels == 2:
            color = 'coral'; component = 'P2'
        for i_int, intensity in enumerate(np.arange(2, 18, 2)):   
            df_mean[i_int, i_labels] = np.mean(df[str(i_labels)][str(i_int)])
            e_mod=(np.std(df[str(i_labels)][str(i_int)]))
            plt.errorbar(intensity, np.mean(df[str(i_labels)][str(i_int)]), e_mod, color = color,   linestyle='None', marker='o' , alpha = 0.6 )
        plt.plot(np.arange(2, 18, 2),  df_mean[:, i_labels], label = component, color = color)
    plt.title(f'{title}', weight = 'bold')    
    plt.ylabel('Normalized Amplitude', weight = 'bold') 
    plt.xlabel('Intensities (mA)', weight = 'bold')    
    plt.legend(ncol=3)
    #plt.ylim(bottom=-1, top=1)
    plt.show()
    
    fig.savefig(save_folder  + f'{title}' +'.svg')
    
    
    
def peaks_plot_channel(peaks, labels, title, save_folder):
    from sklearn import preprocessing as pre
    i_C3 = [i for i,v in enumerate(channel_names()) if v == 'C3'][0]
    i_C4 = [i for i,v in enumerate(channel_names()) if v == 'C4'][0]
    i_F3 = [i for i,v in enumerate(channel_names()) if v == 'F3'][0]
    i_F4 = [i for i,v in enumerate(channel_names()) if v == 'F4'][0]
    i_Cz = [i for i,v in enumerate(channel_names()) if v == 'Cz'][0]
    
    
    if title == 'Stroke, left' or 'Healthy Control, left':
        com_ind = [i_C4, i_F4, i_Cz]
    elif title == 'Stroke, right' or 'Healthy Control, right':
        com_ind = [i_C3, i_F3, i_Cz]

    
    peaks_hc_mean_phase =  np.mean(peaks, (-2))
    df = {}
    for i_l, label in enumerate(labels):
        df[str(i_l)] = {}
        for i_int, intensity in enumerate(np.arange(2, 18, 2)):    
             
             df[str(i_l)][str(i_int)] = pre.MinMaxScaler().fit_transform(peaks_hc_mean_phase[:, com_ind[i_l],i_l, i_int ].reshape(-1, 1))
            
    fig = plt.figure()
    
    df_mean = np.zeros([len((np.arange(2, 18, 2))), len(labels)])
    for i_labels, name_labels in enumerate(labels):
        if  i_labels == 0 :
            color = 'maroon'; component = 'P1'
        elif i_labels == 1:
            color = 'navy'; component = 'N1'
        elif i_labels == 2:
            color = 'coral'; component = 'P2'
        for i_int, intensity in enumerate(np.arange(2, 18, 2)):   
            df_mean[i_int, i_labels] = np.mean(df[str(i_labels)][str(i_int)])
            e_mod=(np.std(df[str(i_labels)][str(i_int)]))
            plt.errorbar(intensity, np.mean(df[str(i_labels)][str(i_int)]), e_mod, color = color,   linestyle='None', marker='o' , alpha = 0.6 )
        plt.plot(np.arange(2, 18, 2),  df_mean[:, i_labels], label = component, color = color)
    plt.title(f'{title}', weight = 'bold')    
    plt.ylabel('Normalized Amplitude ', weight = 'bold') 
    plt.xlabel('Intensities (mA)', weight = 'bold')    
    plt.legend(ncol=3)
    #plt.ylim(bottom=-1, top=1)
    plt.show()
    
    fig.savefig(save_folder  + f'{title}' +'.svg')
    
    
def plot_three_evoked_potentials(Evoked_GrandAv, win_erps, tittle, save_folder_peak):

    fig, sps = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
    fig.suptitle(f'{tittle}', fontsize = 14)
    for i, len_window in enumerate(win_erps):
        if i == 0:
            im = topoplot_2d(Evoked_GrandAv.info['ch_names'], np.max(Evoked_GrandAv. data[:, len_window[0]:len_window[1]], axis =1), Evoked_GrandAv.info, clim=[-2, 2], axes=sps[i], mask=None, maskparam=None)
        if i == 1:
            im = topoplot_2d(Evoked_GrandAv.info['ch_names'], np.min(Evoked_GrandAv. data[:, len_window[0]:len_window[1]], axis =1), Evoked_GrandAv.info, clim=[-2, 2], axes=sps[i], mask=None, maskparam=None)
        if i == 2:
            im = topoplot_2d(Evoked_GrandAv.info['ch_names'], np.max(Evoked_GrandAv. data[:, len_window[0]:len_window[1]], axis =1), Evoked_GrandAv.info, clim=[-2, 2], axes=sps[i], mask=None, maskparam=None)
    
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    cb.set_label('µV', rotation = 90)
    sps[0].set_title('\n\n P1' , fontsize=14, fontweight ='bold')
    sps[1].set_title('\n\n N1' , fontsize=14, fontweight ='bold')
    sps[2].set_title('\n\n P2', fontsize=14, fontweight ='bold')
    fig.savefig(save_folder_peak  + f'{tittle}' +'.svg')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def channel_indices(ch_names):
    ch_inds = []
    for ch in ch_names:
        ch_ind = [i for i,v in enumerate(channel_names()) if v == ch][0]
        ch_inds.append(ch_ind)
    return(ch_inds)



def sub_indices(ch_names):
    ch_inds = []
    for ch in ch_names:
        ch_ind = [i for i,v in enumerate(['AmWo', 'FuMa', 'GrMa', 'KaBe', 'SoFa', 'WiLu', 'BuUl', 'EiHe', 'GuWi', 'MeRu']) if v == ch][0]
        ch_inds.append(ch_ind)
    return(ch_inds)








def Williamson_fig_2_3(contra_left, contra_right, HC_L, HC_R, ST_V2_L, ST_V2_R, save_folder_peak):
    contra_left_ind = channel_indices(contra_left)
    contra_right_ind = channel_indices(contra_right)
    
    fig = plt.figure()
    plt.plot(ST_V2_L[0].times, np.mean(ST_V2_L[0]._data[contra_left_ind,:].T, axis =1), label = 'Stroke Left (Paretic)', color = 'r')
    plt.plot(HC_L[0].times, np.mean(HC_L[0]._data[contra_left_ind,:].T, axis =1), label = 'Control Group Left', color = 'k')
    plt.title('Contralateral Response')
    plt.ylim(-1.5,1.5)
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()
    y_HC_L = np.mean(HC_L[0]._data[contra_left_ind,:].T, axis =1)
    find_peaks(y_HC_L, height=0.7)
    fig.savefig(save_folder_peak + 'L_Contra.svg') 
    
    fig = plt.figure()
    plt.plot(ST_V2_L[0].times, np.mean(ST_V2_L[0]._data[contra_right_ind,:].T, axis =1), label = 'Stroke Left (Paretic)', color = 'r')
    plt.plot(HC_L[0].times, np.mean(HC_L[0]._data[contra_right_ind,:].T, axis =1), label = 'Control Group Left', color = 'k')
    plt.title('Ipsilateral Response')
    plt.ylim(-1.5,1.5)
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'L_Ipsi.svg') 
    
    fig  = plt.figure()
    plt.plot(ST_V2_R[0].times, np.mean(ST_V2_R[0]._data[contra_right_ind,:].T, axis =1), label = 'Stroke Right (Paretic)', color = 'r')
    plt.plot(HC_R[0].times, np.mean(HC_R[0]._data[contra_right_ind,:].T, axis =1), label = 'Control Group Right', color = 'k')
    plt.title('Contralateral Response')
    plt.ylim(-1.5,1.5)
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'R_Contra.svg') 
    
    fig = plt.figure()
    plt.plot(ST_V2_R[0].times, np.mean(ST_V2_R[0]._data[contra_left_ind,:].T, axis =1), label = 'Stroke Right (Paretic)', color = 'r')
    plt.plot(HC_R[0].times, np.mean(HC_R[0]._data[contra_left_ind,:].T, axis =1), label = 'Control Group Right', color = 'k')
    plt.title('Ipsilateral Response')
    plt.ylim(-1.5,1.5)
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'R_Ipsi.svg') 
    
    
    
    
    
    
def amp_latency_6_components_st_deleted_subjects(evokeds_all_L, evokeds_all_R, contra_right, contra_left, time_windows, save_folder_peak):    
    amp_r = {}
    lat_r = {}
    amp_l = {}
    lat_l = {}
    amp_lat_r = {}
    amp_lat_l = {}
    
    time_points = ['v2', 'v3', 'v4', 'v5', 'v6']       
    for i_r,_ in enumerate(range(len(evokeds_all_R[str('v2')]))):
        
        amp_r[str(i_r)] = np.zeros([len(time_points),len(time_windows)])
        lat_r[str(i_r)] = np.zeros([len(time_points),len(time_windows)])
    
        if i_r== 0:
            title_r = 'BuUl(R)_ipsi'; i_corrected = [3, 0, 3, 0, 0]
        elif i_r== 1:
            title_r = 'EiHe(R)_contra'; i_corrected = [0, 2, 1, 1, 2]
        elif i_r==2:
            title_r = 'GuWi(R)_ipsi'; i_corrected = [2, 1, 2, 3, 3]
        elif i_r ==3:
            title_r = 'MeRu(R)_contra'; i_corrected = [1, 3, 0, 2, 1]
            
            
        
        
    
        amp_lat_r[str(i_r)] = {}
        for i_time_point, time_point in enumerate(time_points):

            
            amp_lat_r[str(i_r)][str(time_point)] = {}
            if time_point == 'v2':
                x = 0; y = 0 ; title = f'{time_point}'
            elif time_point == 'v3':
                x = 0; y = 1 ; title = f'{time_point}'
            elif time_point == 'v4':
                x = 0; y = 2 ; title = f'{time_point}'
            elif time_point == 'v5':
                x = 1; y = 0 ; title = f'{time_point}'
            elif time_point == 'v6':
                x = 1; y = 1 ; title = f'{time_point}'
            
            for i_contra_right, n_contra_right in enumerate(contra_right):  

    
                amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)] = np.zeros([len(contra_right), 1001])
                contra_right_ind = channel_indices(contra_right[str(n_contra_right)])
                a = zscore(evokeds_all_R[str(time_point)][i_corrected[i_time_point]].data[:, ], axis =0)
                amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)] = np.mean(a[contra_right_ind, 1000:], axis = 0) 
                
                
                if i_contra_right == 0:
                    color = 'g'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                    
                elif i_contra_right == 1:
                    color = 'g'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.min(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                elif i_contra_right == 2:
                    color = 'navy'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.min(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                    
                elif i_contra_right == 3:
                    color = 'maroon'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                elif i_contra_right == 4:
                    color = 'royalblue'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                    

    
    
    
    #############################
    # Left Side
    
    for i_l,_ in enumerate(range(len(evokeds_all_L[str('v2')]))):
        
        amp_l[str(i_l)] = np.zeros([len(time_points),len(time_windows)])
        lat_l[str(i_l)] = np.zeros([len(time_points),len(time_windows)])
    
            
        if   i_l == 0:
            title_l = 'AmWo(L)_contra'; i_corrected = [2, 5, 4, 4, 4]
        elif i_l == 1:
            title_l = 'FuMa(L)_ipsi'; i_corrected = [3, 0, 5, 5, 5]
        elif i_l == 2:
            title_l = 'GrMa(L)_ipsi'; i_corrected = [4, 2, 3, 2, 3]
        elif i_l == 3:
            title_l = 'KaBe(L)_contra'; i_corrected = [0, 4, 0, 3, 1]
        elif i_l == 4:
            title_l = 'SoFa(L)_contra'; i_corrected = [5, 3, 2, 0, 2]
        elif i_l == 5:
            title_l = 'WiLu(L)_ipsi'; i_corrected = [1, 1, 1, 1, 0]
        
        
    
        amp_lat_l[str(i_l)] = {}
        for i_time_point, time_point in enumerate(time_points):
            
            amp_lat_l[str(i_l)][str(time_point)] = {}
            if time_point == 'v2':
                x = 0; y = 0 ; title = f'{time_point}'
            elif time_point == 'v3':
                x = 0; y = 1 ; title = f'{time_point}'
            elif time_point == 'v4':
                x = 0; y = 2 ; title = f'{time_point}'
            elif time_point == 'v5':
                x = 1; y = 0 ; title = f'{time_point}'
            elif time_point == 'v6':
                x = 1; y = 1 ; title = f'{time_point}'
            
            for i_contra_left, n_contra_left in enumerate(contra_left):  
                
    
                amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)] = np.zeros([len(contra_left), 1001])
                contra_left_ind = channel_indices(contra_left[str(n_contra_left)])
                a = zscore(evokeds_all_L[str(time_point)][i_corrected[i_time_point]].data[:, ], axis =0)
                amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)] = np.mean(evokeds_all_L[str(time_point)][i_corrected[i_time_point]].data[contra_left_ind, 1000:], axis = 0) 
        
                
    
                
                if i_contra_left == 0:
                    color = 'g'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                    
                elif i_contra_left == 1:
                    color = 'g'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.min(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                elif i_contra_left == 2:
                    color = 'navy'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.min(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                    
                elif i_contra_left == 3:
                    color = 'maroon'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                elif i_contra_left == 4:
                    color = 'royalblue'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    

    
    ### 
    
    
    lat_t_std = np.zeros([len(lat_l[str(0)]),len(time_windows)])        
    lat_t_m = (lat_l[str(0)]+ lat_l[str(1)]+ lat_l[str(2)]+ lat_l[str(3)]+lat_l[str(4)]+ lat_l[str(5)] + lat_r[str(0)] + lat_r[str(1)] + lat_r[str(2)] + lat_r[str(3)])/10   
    for i_components, n_components in enumerate(time_windows):  
        lat_t_std[:, i_components] = np.std((lat_l[str(0)][:, i_components],lat_l[str(1)][:, i_components], lat_l[str(2)][:, i_components], lat_l[str(3)][:, i_components], lat_l[str(4)][:, i_components], lat_l[str(5)][:, i_components], lat_r[str(0)][:, i_components], lat_r[str(1)][:, i_components], lat_r[str(2)][:, i_components], lat_r[str(3)][:, i_components]), axis=0)        
    
    fig = plt.figure()    
    for i_components, n_components in enumerate(time_windows): 
       
        if i_components == 0:
            color = 'b'
            
        elif i_components == 1:
            color = 'r'
            
        elif i_components == 2:
            color = 'navy'
            
        elif i_components == 3:
            color = 'maroon'
         
        elif i_components == 4:
            color = 'royalblue'
                         
        elif i_components == 5:
                color = 'crimson' 
        plt.plot(range(len(time_points)),  lat_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
        plt.errorbar(range(len(time_points)),  lat_t_m[:,i_components], lat_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
        plt.xticks(range(len(time_points)), time_points, fontsize=12)
        plt.xlabel('Time points')
        plt.ylabel('Latency')
        plt.title('Both sides')
        plt.legend()
        plt.show()
        
    
        
    fig.savefig(save_folder_peak +  'Latency_6components' + '.svg', overwrite = True) 
    
        
    
        
        
    amp_t_std = np.zeros([len(lat_r[str(0)]),len(time_windows)])        
    amp_t_m= np.zeros([len(lat_r[str(0)]),len(time_windows)])        
    
    for i_components, n_components in enumerate(time_windows):  
        amp_t_std[:, i_components] =np.std((amp_l[str(0)][:, i_components], amp_l[str(1)][:, i_components], amp_l[str(2)][:, i_components], amp_l[str(3)][:, i_components], amp_l[str(4)][:, i_components], amp_l[str(5)][:, i_components], amp_r[str(0)][:, i_components],amp_r[str(1)][:, i_components], amp_r[str(2)][:, i_components], amp_r[str(3)][:, i_components]), axis=0)        
        amp_t_m[:, i_components] = np.mean((amp_l[str(0)][:, i_components], amp_l[str(1)][:, i_components], amp_l[str(2)][:, i_components], amp_l[str(3)][:, i_components], amp_l[str(4)][:, i_components], amp_l[str(5)][:, i_components], amp_r[str(0)][:, i_components],amp_r[str(1)][:, i_components], amp_r[str(2)][:, i_components], amp_r[str(3)][:, i_components]), axis=0)             
    fig = plt.figure()    
    for i_components, n_components in enumerate(time_windows): 
       
        if i_components == 0:
            color = 'b'
            
        elif i_components == 1:
            color = 'r'
            
        elif i_components == 2:
            color = 'navy'
            
        elif i_components == 3:
            color = 'maroon'
         
        elif i_components == 4:
            color = 'royalblue'
                         
        elif i_components == 5:
                color = 'crimson' 
        plt.plot(range(len(time_points)),  amp_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
        plt.errorbar(range(len(time_points)),  amp_t_m[:,i_components], amp_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
        plt.xticks(range(len(time_points)), time_points, fontsize=12)
        plt.xlabel('Time points')
        plt.ylabel('Amplitude')
        plt.title('Both Sides')
        plt.legend()
        plt.show()
        
        
        
    fig = plt.figure()    
    i_components == 5; color = 'crimson' 

    plt.plot(range(len(time_points)),  amp_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
    plt.errorbar(range(len(time_points)),  amp_t_m[:,i_components], amp_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
    plt.xticks(range(len(time_points)), time_points, fontsize=12)
    plt.xlabel('Time points')
    plt.ylabel('Amplitude')
    plt.title('Both Sides')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak +  'Amplitude_6components' + '.svg', overwrite = True) 
    
    
    amp_l_arr = np.zeros([6, 6, 5])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            amp_l_arr[i_components,i,:] = amp_l[str(i)][:,i_components]
    
    
    lat_l_arr = np.zeros([6, 6, 5])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            lat_l_arr[i_components,i,:] = lat_l[str(i)][:,i_components]
    
    
    amp_r_arr = np.zeros([6, 4, 5])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            amp_r_arr[ i_components,i,:] = amp_r[str(i)][:,i_components]
            
    lat_r_arr = np.zeros([6, 4, 5])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            lat_r_arr[ i_components,i,:] = lat_r[str(i)][:,i_components]
            
    return(amp_l_arr, amp_r_arr, lat_l_arr, lat_r_arr) 
    
    
    
    
    
    
    
    

    
def amp_latency_6_components_st(evokeds_all_L, evokeds_all_R, contra_right, contra_left, time_windows, save_folder_peak):    
    amp_r = {}
    lat_r = {}
    amp_l = {}
    lat_l = {}
    amp_lat_r = {}
    amp_lat_l = {}
    
    time_points = ['v2', 'v3', 'v4', 'v5', 'v6']       
    for i_r,_ in enumerate(range(len(evokeds_all_R[str('v2')]))):
        
        amp_r[str(i_r)] = np.zeros([len(time_points),len(time_windows)])
        lat_r[str(i_r)] = np.zeros([len(time_points),len(time_windows)])
    
        if i_r== 0:
            title_r = 'BuUl(R)_ipsi'; i_corrected = [3, 0, 3, 0, 0]
        elif i_r== 1:
            title_r = 'EiHe(R)_contra'; i_corrected = [0, 2, 1, 1, 2]
        elif i_r==2:
            title_r = 'GuWi(R)_ipsi'; i_corrected = [2, 1, 2, 3, 3]
        elif i_r ==3:
            title_r = 'MeRu(R)_contra'; i_corrected = [1, 3, 0, 2, 1]
            
            
        
        
    
        amp_lat_r[str(i_r)] = {}
        for i_time_point, time_point in enumerate(time_points):

            
            amp_lat_r[str(i_r)][str(time_point)] = {}
            if time_point == 'v2':
                x = 0; y = 0 ; title = f'{time_point}'
            elif time_point == 'v3':
                x = 0; y = 1 ; title = f'{time_point}'
            elif time_point == 'v4':
                x = 0; y = 2 ; title = f'{time_point}'
            elif time_point == 'v5':
                x = 1; y = 0 ; title = f'{time_point}'
            elif time_point == 'v6':
                x = 1; y = 1 ; title = f'{time_point}'
            
            for i_contra_right, n_contra_right in enumerate(contra_right):  

    
                amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)] = np.zeros([len(contra_right), 1001])
                contra_right_ind = channel_indices(contra_right[str(n_contra_right)])
                a = zscore(evokeds_all_R[str(time_point)][i_corrected[i_time_point]].data[:, ], axis =0)
                amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)] = np.mean(a[contra_right_ind, 1000:], axis = 0) 
                
                
                if i_contra_right == 0:
                    color = 'g'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                    
                elif i_contra_right == 1:
                    color = 'g'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.min(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                elif i_contra_right == 2:
                    color = 'navy'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.min(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                    
                elif i_contra_right == 3:
                    color = 'maroon'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                elif i_contra_right == 4:
                    color = 'royalblue'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    

    
    
    
    #############################
    # Left Side
    
    for i_l,_ in enumerate(range(len(evokeds_all_L[str('v2')]))):
        
        amp_l[str(i_l)] = np.zeros([len(time_points),len(time_windows)])
        lat_l[str(i_l)] = np.zeros([len(time_points),len(time_windows)])
    
            
        if   i_l == 0:
            title_l = 'AmWo(L)_contra'; i_corrected = [2, 5, 4, 4, 4]
        elif i_l == 1:
            title_l = 'FuMa(L)_ipsi'; i_corrected = [3, 0, 5, 5, 5]
        elif i_l == 2:
            title_l = 'GrMa(L)_ipsi'; i_corrected = [4, 2, 3, 2, 3]
        elif i_l == 3:
            title_l = 'KaBe(L)_contra'; i_corrected = [0, 4, 0, 3, 1]
        elif i_l == 4:
            title_l = 'SoFa(L)_contra'; i_corrected = [5, 3, 2, 0, 2]
        elif i_l == 5:
            title_l = 'WiLu(L)_ipsi'; i_corrected = [1, 1, 1, 1, 0]
        
        
    
        amp_lat_l[str(i_l)] = {}
        for i_time_point, time_point in enumerate(time_points):
            
            amp_lat_l[str(i_l)][str(time_point)] = {}
            if time_point == 'v2':
                x = 0; y = 0 ; title = f'{time_point}'
            elif time_point == 'v3':
                x = 0; y = 1 ; title = f'{time_point}'
            elif time_point == 'v4':
                x = 0; y = 2 ; title = f'{time_point}'
            elif time_point == 'v5':
                x = 1; y = 0 ; title = f'{time_point}'
            elif time_point == 'v6':
                x = 1; y = 1 ; title = f'{time_point}'
            
            for i_contra_left, n_contra_left in enumerate(contra_left):  
                
    
                amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)] = np.zeros([len(contra_left), 1001])
                contra_left_ind = channel_indices(contra_left[str(n_contra_left)])
                a = zscore(evokeds_all_L[str(time_point)][i_corrected[i_time_point]].data[:, ], axis =0)
                amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)] = np.mean(evokeds_all_L[str(time_point)][i_corrected[i_time_point]].data[contra_left_ind, 1000:], axis = 0) 
        
                
    
                
                if i_contra_left == 0:
                    color = 'g'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                    
                elif i_contra_left == 1:
                    color = 'g'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.min(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                elif i_contra_left == 2:
                    color = 'navy'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.min(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                    
                elif i_contra_left == 3:
                    color = 'maroon'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                elif i_contra_left == 4:
                    color = 'royalblue'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    

    
    
    ### 
    
    
    lat_t_std = np.zeros([len(lat_l[str(0)]),len(time_windows)])        
    lat_t_m = (lat_l[str(0)]+ lat_l[str(1)]+ lat_l[str(2)]+ lat_l[str(3)]+lat_l[str(4)]+ lat_l[str(5)] + lat_r[str(0)] + lat_r[str(1)] + lat_r[str(2)] + lat_r[str(3)])/10   
    for i_components, n_components in enumerate(time_windows):  
        lat_t_std[:, i_components] = np.std((lat_l[str(0)][:, i_components],lat_l[str(1)][:, i_components], lat_l[str(2)][:, i_components], lat_l[str(3)][:, i_components], lat_l[str(4)][:, i_components], lat_l[str(5)][:, i_components], lat_r[str(0)][:, i_components], lat_r[str(1)][:, i_components], lat_r[str(2)][:, i_components], lat_r[str(3)][:, i_components]), axis=0)        
    
    fig = plt.figure()    
    for i_components, n_components in enumerate(time_windows): 
       
        if i_components == 0:
            color = 'b'
            
        elif i_components == 1:
            color = 'r'
            
        elif i_components == 2:
            color = 'navy'
            
        elif i_components == 3:
            color = 'maroon'
         
        elif i_components == 4:
            color = 'royalblue'
                         
 
        plt.plot(range(len(time_points)),  lat_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
        plt.errorbar(range(len(time_points)),  lat_t_m[:,i_components], lat_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
        plt.xticks(range(len(time_points)), time_points, fontsize=12)
        plt.xlabel('Time points')
        plt.ylabel('Latency')
        plt.title('Both sides')
        plt.legend()
        plt.show()
        
    
        
    fig.savefig(save_folder_peak +  'Latency_6components' + '.svg', overwrite = True) 
    
        
    
        
        
    amp_t_std = np.zeros([len(lat_r[str(0)]),len(time_windows)])        
    amp_t_m= np.zeros([len(lat_r[str(0)]),len(time_windows)])        
    
    for i_components, n_components in enumerate(time_windows):  
        amp_t_std[:, i_components] =np.std((amp_l[str(0)][:, i_components], amp_l[str(1)][:, i_components], amp_l[str(2)][:, i_components], amp_l[str(3)][:, i_components], amp_l[str(4)][:, i_components], amp_l[str(5)][:, i_components], amp_r[str(0)][:, i_components],amp_r[str(1)][:, i_components], amp_r[str(2)][:, i_components], amp_r[str(3)][:, i_components]), axis=0)        
        amp_t_m[:, i_components] = np.mean((amp_l[str(0)][:, i_components], amp_l[str(1)][:, i_components], amp_l[str(2)][:, i_components], amp_l[str(3)][:, i_components], amp_l[str(4)][:, i_components], amp_l[str(5)][:, i_components], amp_r[str(0)][:, i_components],amp_r[str(1)][:, i_components], amp_r[str(2)][:, i_components], amp_r[str(3)][:, i_components]), axis=0)             
    fig = plt.figure()    
    for i_components, n_components in enumerate(time_windows): 
       
        if i_components == 0:
            color = 'b'
            
        elif i_components == 1:
            color = 'r'
            
        elif i_components == 2:
            color = 'navy'
            
        elif i_components == 3:
            color = 'maroon'
         
        elif i_components == 4:
            color = 'royalblue'
                         

        plt.plot(range(len(time_points)),  amp_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
        plt.errorbar(range(len(time_points)),  amp_t_m[:,i_components], amp_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
        plt.xticks(range(len(time_points)), time_points, fontsize=12)
        plt.xlabel('Time points')
        plt.ylabel('Amplitude')
        plt.title('Both Sides')
        plt.legend()
        plt.show()
        
        
        
    fig = plt.figure()    


    plt.plot(range(len(time_points)),  amp_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
    plt.errorbar(range(len(time_points)),  amp_t_m[:,i_components], amp_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
    plt.xticks(range(len(time_points)), time_points, fontsize=12)
    plt.xlabel('Time points')
    plt.ylabel('Amplitude')
    plt.title('Both Sides')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak +  'Amplitude_6components' + '.svg', overwrite = True) 
    
    
    amp_l_arr = np.zeros([5, 6, 5])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            amp_l_arr[i_components,i,:] = amp_l[str(i)][:,i_components]
    
    
    lat_l_arr = np.zeros([5, 6, 5])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            lat_l_arr[i_components,i,:] = lat_l[str(i)][:,i_components]
    
    
    amp_r_arr = np.zeros([5, 4, 5])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            amp_r_arr[ i_components,i,:] = amp_r[str(i)][:,i_components]
            
    lat_r_arr = np.zeros([5, 4, 5])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            lat_r_arr[ i_components,i,:] = lat_r[str(i)][:,i_components]
            
    return(amp_l_arr, amp_r_arr, lat_l_arr, lat_r_arr)        




    
    
    

    
def amp_latency_6_components_hc(evokeds_all_L_hc, evokeds_all_R_hc, contra_right, contra_left, time_windows, save_folder_peak):    
    amp_r = {}
    lat_r = {}
    amp_l = {}
    lat_l = {}
    amp_lat_r = {}
    amp_lat_l = {}
    
   
    for i_r,_ in enumerate(range(len(evokeds_all_R_hc))):
        amp_lat_r[str(i_r)] = {}
        amp_r[str(i_r)] = np.zeros(len(time_windows))
        lat_r[str(i_r)] = np.zeros(len(time_windows))
        
        for i_contra_right, n_contra_right in enumerate(contra_right):  
            amp_lat_r[str(i_r)][str(n_contra_right)] = np.zeros([len(contra_right), 1001])
            contra_right_ind = channel_indices(contra_right[str(n_contra_right)])
            a = zscore(evokeds_all_R_hc[i_r].data[:, ], axis =0)
            a = (evokeds_all_R_hc[i_r].data[:, ])
            amp_lat_r[str(i_r)][str(n_contra_right)] = np.mean(a[contra_right_ind, 1000:], axis = 0) 
            
            
            if i_contra_right == 0:
                color = 'g'
                amp_r[str(i_r)][i_contra_right] = np.max(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]

                
            elif i_contra_right == 1:
                color = 'g'
                amp_r[str(i_r)][i_contra_right] = np.min(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]

            elif i_contra_right == 2:
                color = 'navy'
                amp_r[str(i_r)][i_contra_right] = np.min(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]

                
            elif i_contra_right == 3:
                color = 'maroon'
                amp_r[str(i_r)][i_contra_right] = np.max(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]

            elif i_contra_right == 4:
                color = 'royalblue'
                amp_r[str(i_r)][i_contra_right] = np.max(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]


    
    
    
    #############################
    # Left Side
    
    for i_l,_ in enumerate(range(len(evokeds_all_L_hc))):
        amp_lat_l[str(i_l)] = {}
        amp_l[str(i_l)] = np.zeros(len(time_windows))
        lat_l[str(i_l)] = np.zeros(len(time_windows))

            
        for i_contra_left, n_contra_left in enumerate(contra_left):  
            amp_lat_l[str(i_l)][str(n_contra_left)] = np.zeros([len(contra_left), 1001])
            contra_left_ind = channel_indices(contra_left[str(n_contra_left)])
            a = zscore(evokeds_all_L_hc[i_l].data[:, ], axis =0)
            amp_lat_l[str(i_l)][str(n_contra_left)] = np.mean(evokeds_all_L_hc[i_l].data[contra_left_ind, 1000:], axis = 0) 
    
            

            
            if i_contra_left == 0:
                color = 'g'
                amp_l[str(i_l)][i_contra_left] = np.min(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

                
            elif i_contra_left == 1:
                color = 'g'
                amp_l[str(i_l)][i_contra_left] = np.max(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

            elif i_contra_left == 2:
                color = 'navy'
                amp_l[str(i_l)][i_contra_left] = np.min(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

                
            elif i_contra_left == 3:
                color = 'maroon'
                amp_l[str(i_l)][i_contra_left] = np.max(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

            elif i_contra_left == 4:
                color = 'royalblue'
                amp_l[str(i_l)][i_contra_left] = np.min(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

                

    
    
    ### 
    
    
    amp_l_arr_hc = np.zeros([5, 11])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            amp_l_arr_hc[i_components,i] = amp_l[str(i)][i_components]
    
    
    lat_l_arr_hc = np.zeros([5, 11])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            lat_l_arr_hc[i_components,i] = lat_l[str(i)][i_components]
    
    
    amp_r_arr_hc = np.zeros([5, 6])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            amp_r_arr_hc[i_components,i] = amp_r[str(i)][i_components]
            
    lat_r_arr_hc = np.zeros([5, 6])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            lat_r_arr_hc[i_components,i] = lat_r[str(i)][i_components]
    
    
    
    
    amp_t = np.concatenate((amp_l_arr_hc, amp_r_arr_hc), axis = 1)
    lat_t = np.concatenate((lat_l_arr_hc, lat_r_arr_hc), axis = 1)
    


            
    return(amp_t, lat_t)        




def box_plot_n2_HC_st(lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):
    dict_n2 = {'Control': lat_t_hc[2, :], 'Stroke V2': np.concatenate((lat_r_arr_st[2, :, 0], lat_l_arr_st[2, :, 0])),
                                          'Stroke V3': np.concatenate((lat_r_arr_st[2, :, 1], lat_l_arr_st[2, :, 1])),
                                          'Stroke V4': np.concatenate((lat_r_arr_st[2, :, 2], lat_l_arr_st[2, :, 2])),
                                          'Stroke V5': np.concatenate((lat_r_arr_st[2, :, 3], lat_l_arr_st[2, :, 3])), 
                                          'Stroke V6': np.concatenate((lat_r_arr_st[2, :, 4], lat_l_arr_st[2, :, 4]))}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(dict_n2.values(), showfliers=False)
    palette = ['k','r', 'orange', 'b', 'y', 'g']
    for i, val, c in zip(np.arange(0,6), dict_n2.values(), palette):
        plt.scatter(np.repeat(i+1, len(val)), val, alpha=0.4, color=c)   
    ax.set_xticklabels(dict_n2.keys()) 
    ax.set_ylabel('Latency (ms)')
    ax.set_title('N2 Latency', fontweight="bold")
    plt.show()
    fig.savefig(save_folder_peak  + 'n2_latency_boxplot'+ '.svg', overwrite = True)
    
    
    
    
def bar_plot_n2_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):

    dict_n2 = {'Control': lat_t_hc[component_number, :], 'Stroke V2': np.concatenate((lat_r_arr_st[component_number, :, 0], lat_l_arr_st[component_number, :, 0])),
                                          'Stroke V3': np.concatenate((lat_r_arr_st[component_number, :, 1], lat_l_arr_st[component_number, :, 1])),
                                          'Stroke V4': np.concatenate((lat_r_arr_st[component_number, :, 2], lat_l_arr_st[component_number, :, 2])),
                                          'Stroke V5': np.concatenate((lat_r_arr_st[component_number, :, 3], lat_l_arr_st[component_number, :, 3])), 
                                          'Stroke V6': np.concatenate((lat_r_arr_st[component_number, :, 4], lat_l_arr_st[component_number, :, 4]))}
    fig, ax = plt.subplots()
    data_std = np.zeros([1, 6])
    data_mean =  np.zeros([1, 6])
    data_std = [np.std(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    data_mean = [np.mean(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    ax.plot(np.array([1, 2, 3, 4, 5]), data_mean[1:], color = 'k')
    ax.scatter(.35, 93, marker = '*', color = 'k')
    ax.plot([0, 1], [90, 90], color = 'k')
    ax.plot([0, 0, 0, 0], [90, 89, 88, 87], color = 'k')
    ax.plot([1, 1, 1, 1], [90, 89, 88, 87], color = 'k')
    ax.scatter(3, 103, marker = '*', color = 'k')
    ax.plot([1, 2, 3, 4, 5], [100, 100, 100, 100, 100], color = 'k')
    ax.plot([1, 1, 1, 1], [100, 99, 98, 97], color = 'k')
    ax.plot([5, 5, 5, 5], [100, 99, 98, 97], color = 'k')
    ax.bar( x = np.arange(len(dict_n2.keys())), height = data_mean, yerr= data_std, capsize=4, color ='grey', alpha = 0.7)
    ax.set_xticklabels(['Control','Control', 'Stroke V2', 'Stroke V3', 'Stroke V4', 'Stroke V5', 'Stroke V6']) 
    ax.set_ylabel('Latency (ms)')
    ax.set_title('N1 Latency', fontweight="bold")
    ax.set_ylim([0, 130])
    t_ind , p_ind = stats.ttest_ind(dict_n2[str('Control')], dict_n2[str('Stroke V2')])
    t_paired, p_paired = stats.ttest_rel(dict_n2[str('Stroke V2')], dict_n2[str('Stroke V6')])
    ax.text(-0.3 , 120, f'P-value = {np.round(p_ind, 3)}', fontweight = 'bold')
    ax.text(2 , 120, f'P-value = {np.round(p_paired, 3)}', fontweight = 'bold')
    plt.show()
    fig.savefig(save_folder_peak  + 'n2_latency_barplot'+ '.svg', overwrite = True)
    

    
    return(t_ind , p_ind, t_paired, p_paired)








    
    
    
def bar_plot_p2_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):

    dict_n2 = {'Control': lat_t_hc[component_number, :], 'Stroke V2': np.concatenate((lat_r_arr_st[component_number, :, 0], lat_l_arr_st[component_number, :, 0])),
                                          'Stroke V3': np.concatenate((lat_r_arr_st[component_number, :, 1], lat_l_arr_st[component_number, :, 1])),
                                          'Stroke V4': np.concatenate((lat_r_arr_st[component_number, :, 2], lat_l_arr_st[component_number, :, 2])),
                                          'Stroke V5': np.concatenate((lat_r_arr_st[component_number, :, 3], lat_l_arr_st[component_number, :, 3])), 
                                          'Stroke V6': np.concatenate((lat_r_arr_st[component_number, :, 4], lat_l_arr_st[component_number, :, 4]))}
    fig, ax = plt.subplots()
    data_std = np.zeros([1, 6])
    data_mean =  np.zeros([1, 6])
    data_std = [np.std(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    data_mean = [np.mean(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    ax.plot(np.array([1, 2, 3, 4, 5]), data_mean[1:], color = 'k')
    ax.plot([0, 1], [220, 220], color = 'k')
    ax.plot([0, 0, 0, 0], [219, 218, 217, 216], color = 'k')
    ax.plot([1, 1, 1, 1], [219, 218, 217, 216], color = 'k')
    ax.plot([1, 2, 3, 4, 5], [230, 230, 230, 230, 230], color = 'k')
    ax.plot([1, 1, 1, 1], [229, 228, 227, 226], color = 'k')
    ax.plot([5, 5, 5, 5], [229, 228, 227, 226], color = 'k')
    ax.bar( x = np.arange(len(dict_n2.keys())), height = data_mean, yerr= data_std, capsize=4, color ='grey', alpha = 0.7)
    ax.set_xticklabels(['Control','Control', 'Stroke V2', 'Stroke V3', 'Stroke V4', 'Stroke V5', 'Stroke V6']) 
    ax.set_ylabel('Latency (ms)')
    ax.set_title('P2 Latency', fontweight="bold")
    ax.set_ylim([0, 260])
    t_ind , p_ind = stats.ttest_ind(dict_n2[str('Control')], dict_n2[str('Stroke V2')])
    t_paired, p_paired = stats.ttest_rel(dict_n2[str('Stroke V2')], dict_n2[str('Stroke V6')])
    ax.text(-0.3 , 240, f'P-value = {np.round(p_ind, 3)}', fontweight = 'bold')
    ax.text(2 , 240, f'P-value = {np.round(p_paired, 3)}', fontweight = 'bold')
    plt.show()
    fig.savefig(save_folder_peak  + 'p2_latency_barplot'+ '.svg', overwrite = True)
    

    
    return(t_ind , p_ind, t_paired, p_paired)








def errorbar_plot_n2_HC_st(lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):
    dict_n2 = {'Control': lat_t_hc[2, :], 'Stroke V2': np.concatenate((lat_r_arr_st[2, :, 0], lat_l_arr_st[2, :, 0])),
                                          'Stroke V3': np.concatenate((lat_r_arr_st[2, :, 1], lat_l_arr_st[2, :, 1])),
                                          'Stroke V4': np.concatenate((lat_r_arr_st[2, :, 2], lat_l_arr_st[2, :, 2])),
                                          'Stroke V5': np.concatenate((lat_r_arr_st[2, :, 3], lat_l_arr_st[2, :, 3])), 
                                          'Stroke V6': np.concatenate((lat_r_arr_st[2, :, 4], lat_l_arr_st[2, :, 4]))}
    
    data_std = np.zeros([1, 6])
    data_mean =  np.zeros([1, 6])
    data_std = [np.std(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    data_mean = [np.mean(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    fig, ax = plt.subplots()
    ax.errorbar( x = np.arange(len(dict_n2.keys())), y = data_mean, yerr= data_std, capsize=4, color ='grey', alpha = 0.7)
    palette = ['k','r', 'orange', 'b', 'y', 'g']
    for i, val, c in zip(np.arange(0,6), dict_n2.values(), palette):
        plt.scatter(np.repeat(i, len(val)), val, alpha=0.4, color=c)   
    ax.set_xticklabels(['Control', 'Control', 'Stroke V2', 'Stroke V3', 'Stroke V4', 'Stroke V5', 'Stroke V6']) 
    ax.set_ylabel('Latency (ms)')
    ax.set_title('N2 Latency', fontweight="bold")
    plt.show()
    fig.savefig(save_folder_peak  + 'n2_latency_errorbarplot'+ '.svg', overwrite = True)
    
    
    

def laterality_Index_errorbar(contra_right, contra_left, evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, save_folder_peak):

    contra_right_ind = channel_indices(contra_right)
    contra_left_ind = channel_indices(contra_left)
    contra_stroke = np.zeros([10, 315]); ipsi_stroke = np.zeros([10, 315])   
    contra_hc = np.zeros([17, 315]); ipsi_hc = np.zeros([17, 315])   
    
    
    # Stroke ERPs
    for i, _ in enumerate(evokeds_all_L[str('v2')]):     
        contra_stroke[i, :] = np.mean(evokeds_all_L[str('v2')][i]._data[contra_left_ind, 985:1300], axis = 0)    
    for i, _ in enumerate(evokeds_all_R[str('v2')]):
        contra_stroke[i+6, :] = np.mean(evokeds_all_R[str('v2')][i]._data[contra_right_ind, 985:1300], axis = 0)
    for i, _ in enumerate(evokeds_all_L[str('v2')]):     
        ipsi_stroke[i, :] = np.mean(evokeds_all_L[str('v2')][i]._data[contra_right_ind, 985:1300], axis = 0)    
    for i, _ in enumerate(evokeds_all_R[str('v2')]):
        ipsi_stroke[i+6, :] = np.mean(evokeds_all_R[str('v2')][i]._data[contra_left_ind, 985:1300], axis = 0)
        
    # Healthy ERPs
    for i, _ in enumerate(evokeds_all_L_hc):     
        contra_hc[i, :] = np.mean(evokeds_all_L_hc[i]._data[contra_left_ind, 985:1300], axis = 0)    
    for i, _ in enumerate(evokeds_all_R_hc):
        contra_hc[i+11, :] = np.mean(evokeds_all_R_hc[i]._data[contra_right_ind, 985:1300], axis = 0)
    for i, _ in enumerate(evokeds_all_L_hc):     
        ipsi_hc[i, :] = np.mean(evokeds_all_L_hc[i]._data[contra_right_ind, 985:1300], axis = 0)    
    for i, _ in enumerate(evokeds_all_R_hc):
        ipsi_hc[i+11, :] = np.mean(evokeds_all_R_hc[i]._data[contra_left_ind, 985:1300], axis = 0)    
        
        
        
    LI_ST = (contra_stroke - ipsi_stroke) / (contra_stroke + ipsi_stroke)  
    LI_HC = (contra_hc - ipsi_hc) / (contra_hc + ipsi_hc)  
    
    
    LI_ST[LI_ST > 50] = 50
    LI_ST[LI_ST < -50] = -50
    
    LI_HC[LI_HC > 50] = 50
    LI_HC[LI_HC < -50] = -50
    
    
    
    
    
    
    fig, ax = plt.subplots()
    ax.set_title('Laterality Index',  fontweight="bold")
    data_std = np.zeros([1, 2])
    data_mean =  np.zeros([1, 2])
    start_time = 115
    end_time = 150
    
    data_mean[0, 0] = np.mean(np.mean(LI_HC[:, start_time:end_time], axis =1), axis =0)
    data_mean[0, 1] = np.mean(np.mean(LI_ST[:, start_time:end_time], axis =1), axis =0)  
    
    data_std[0, 0] = np.std(np.mean(LI_HC[:, start_time:end_time], axis =1), axis =0)/np.sqrt(17)
    data_std[0, 1] = np.std(np.mean(LI_ST[:, start_time:end_time], axis =1), axis =0)/np.sqrt(10)
    
    t_ind , p_ind = stats.ttest_ind(np.mean(LI_HC[:, start_time:end_time], axis =1), np.mean(LI_ST[:, start_time:end_time], axis =1))
    ax.text(0.3, 2.5, f'P-value = {np.round(p_ind, 3)}', fontweight = 'bold')
    ax.scatter(0.5, 2.2, marker = '*', color = 'k')
    ax.plot([0, 1], [2, 2], color = 'k')
    ax.errorbar(x = [0,1], y = data_mean[0], yerr= data_std[0],  color ='k', fmt = '.', capsize=10, capthick=3, elinewidth=2)
    ax.set_xticklabels([  '','', 'Control',  '',   '', '','Stroke V2', ''], fontsize =12)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-3, 3])
    ax.set_ylabel('Laterality Index', fontsize =12)
    plt.show()
    fig.savefig(save_folder_peak  + 'Laterality_Index_errorbar'+ '.svg', overwrite = True)
    return()



  
def ipsi_contra_response_william_2022_fig2(contra_right, contra_left, save_folder_peak):    
    
    contra_right_ind = channel_indices(contra_right)
    contra_left_ind = channel_indices(contra_left)
    with open(str(save_folder_peak) + 'ST_ERP_R_V2_V6.p', 'rb') as fp:
        st_v2_v6_R = pickle.load(fp)
        
    with open(str(save_folder_peak) + 'ST_ERP_L_V2_V6.p', 'rb') as fp:
        st_v2_v6_L = pickle.load(fp)
        
    HC_R = mne.read_evokeds(save_folder_peak + 'HC_R_ave.fif')
    HC_L = mne.read_evokeds(save_folder_peak + 'HC_L_ave.fif')
    
    
    HC_L[0].plot_topomap(np.arange(0.015, 0.04, 0.005), show_names = True, scalings = dict(eeg=1),  sphere=(0.00, 0.00, 0.00, 0.11), vlim=(-2,2))
    HC_R[0].plot_topomap(np.arange(0.015, 0.04, 0.005), show_names = True, scalings = dict(eeg=1),  sphere=(0.00, 0.00, 0.00, 0.11), vlim=(-2,2))
    
    time_points = ['v2', 'v3', 'v4', 'v5', 'v6']
    
    peaks = {}
    peaks_ind = {}
    fig, ax  = plt.subplots()
    for _,time_point in enumerate(time_points):
        st_amp_v2_v6 = np.mean(st_v2_v6_R[str(time_point)]._data[contra_right_ind, :], axis = 0)
        peaks_ind[str(time_point)], peaks[str(time_point)] = find_peaks(st_amp_v2_v6, height = 0.3)
        if   time_point == 'v2':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'r')
        elif time_point == 'v3':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'orange')
        elif time_point == 'v4':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'blue')
        elif time_point == 'v5':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'royalblue')
        elif time_point == 'v6':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'green')
    hc_ind,_  =find_peaks( np.mean(HC_R[0]._data[contra_right_ind, :], axis = 0))    
    ax.plot(HC_R[0].times, np.mean(HC_R[0]._data[contra_right_ind, :], axis = 0), label  = f'HC, {hc_ind[0]-15}ms', color = 'k')
    ax.set_ylim(-2, 1.5)
    ax.set_xlabel('Latency(ms)')
    ax.set_ylabel('Mean Amplitude(µV)')
    plt.title('Contralateral Response to Right Stimulation side')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'R_Contra_v2_v6.svg') 
    
    
    
    fig, ax  = plt.subplots()
    for _,time_point in enumerate(time_points):
        st_amp_v2_v6 = np.mean(st_v2_v6_L[str(time_point)]._data[contra_left_ind, :], axis = 0)
        peaks_ind[str(time_point)], peaks[str(time_point)] = find_peaks(st_amp_v2_v6, height = 0.3)
        if   time_point == 'v2':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'r')
        elif time_point == 'v3':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'orange')
        elif time_point == 'v4':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'blue')
        elif time_point == 'v5':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'royalblue')
        elif time_point == 'v6':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'green')
    hc_ind,_  =find_peaks( np.mean(HC_L[0]._data[contra_left_ind, :], axis = 0)) 
    ax.plot(HC_L[0].times, np.mean(HC_L[0]._data[contra_left_ind, :], axis = 0), label  = f'HC, {hc_ind[0]-15}ms', color = 'k')
    ax.set_ylim(-2, 1.5)
    ax.set_xlabel('Latency(ms)')
    ax.set_ylabel('Mean Amplitude(µV)')
    plt.title('Contralateral Response to Left Stimulation side')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'L_Contra_v2_v6.svg') 
    
    
    
    
    
    
    
    
    time_points = ['v2', 'v6']
    time_points = ['v2']
    fig, ax  = plt.subplots()
    for _,time_point in enumerate(time_points):
        st_amp_v2_v6 = (np.mean(st_v2_v6_L[str(time_point)]._data[contra_left_ind, :], axis = 0)+np.mean(st_v2_v6_R[str(time_point)]._data[contra_right_ind, :], axis = 0))/2
        peaks_ind[str(time_point)], peaks[str(time_point)] = find_peaks(st_amp_v2_v6, height = 0.3)
        if   time_point == 'v2':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = 'S', color = 'r')
        elif time_point == 'v3':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'orange')
        elif time_point == 'v4':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'blue')
        elif time_point == 'v5':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}, {peaks_ind[str(time_point)][0]-15}ms', color = 'royalblue')
        elif time_point == 'v6':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'green')
    hc_ind,_  =find_peaks( (np.mean(HC_L[0]._data[contra_left_ind, :], axis = 0) + np.mean(HC_R[0]._data[contra_right_ind, :], axis = 0))/2) 
    ax.plot(HC_L[0].times, (np.mean(HC_L[0]._data[contra_left_ind, :], axis = 0) + np.mean(HC_R[0]._data[contra_right_ind, :], axis = 0))/2, label  = 'C', color = 'k')
    ax.set_ylim(-2, 1.5)
    ax.set_xlabel('Latency(ms)')
    ax.set_ylabel('Mean amplitude(µV)')
    plt.title('Contralateral response')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'both_Contra_v2_v6.svg') 
    
    
    fig, ax  = plt.subplots()
    for _,time_point in enumerate(time_points):
        st_amp_v2_v6 = (np.mean(st_v2_v6_L[str(time_point)]._data[contra_right_ind, :], axis = 0) + np.mean(st_v2_v6_R[str(time_point)]._data[contra_left_ind, :], axis = 0))/2
        if   time_point == 'v2':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = 'S', color = 'r')
        elif time_point == 'v3':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'orange')
        elif time_point == 'v4':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'blue')
        elif time_point == 'v5':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'royalblue')
        elif time_point == 'v6':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'green')
     
    ax.plot(HC_L[0].times, (np.mean(HC_L[0]._data[contra_right_ind, :], axis = 0) + np.mean(HC_R[0]._data[contra_left_ind, :], axis = 0))/2, label  = 'C', color = 'k')
    ax.set_ylim(-2, 1.5)
    ax.set_xlabel('Latency(ms)')
    ax.set_ylabel('Mean amplitude(µV)')
    plt.title('Ipsilateral response')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'both_Ipsi_v2_v6.svg') 
    
    
    
    
    fig, ax  = plt.subplots()
    for _,time_point in enumerate(time_points):
        st_amp_v2_v6 = np.mean(st_v2_v6_L[str(time_point)]._data[contra_right_ind, :], axis = 0)
        if   time_point == 'v2':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'r')
        elif time_point == 'v3':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'orange')
        elif time_point == 'v4':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'blue')
        elif time_point == 'v5':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'royalblue')
        elif time_point == 'v6':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'green')
     
    ax.plot(HC_L[0].times, np.mean(HC_L[0]._data[contra_right_ind, :], axis = 0), label  = 'HC', color = 'k')
    ax.set_ylim(-2, 1.5)
    ax.set_xlabel('Latency(ms)')
    ax.set_ylabel('Mean Amplitude(µV)')
    plt.title('Ipsilateral Response to Left Stimulation side')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'L_Ipsi_v2_v6.svg') 
    
    
    fig, ax  = plt.subplots()
    for _,time_point in enumerate(time_points):
        st_amp_v2_v6 = np.mean(st_v2_v6_R[str(time_point)]._data[contra_left_ind, :], axis = 0)
        if   time_point == 'v2':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'r')
        elif time_point == 'v3':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'orange')
        elif time_point == 'v4':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'blue')
        elif time_point == 'v5':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'royalblue')
        elif time_point == 'v6':
            ax.plot(st_v2_v6_R[str(time_point)].times, st_amp_v2_v6, label  = f'{time_point}', color = 'green')
     
    ax.plot(HC_R[0].times, np.mean(HC_R[0]._data[contra_left_ind, :], axis = 0), label  = 'HC', color = 'k')
    ax.set_ylim(-2, 1.5)
    ax.set_xlabel('Latency(ms)')
    ax.set_ylabel('Mean Amplitude(µV)')
    plt.title('Ipsilateral Response to Right Stimulation side')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak + 'R_Ipsi_v2_v6.svg')
    
    
    time_point = 'v2'
    fig = plt.figure()
    contra =  ((np.mean(st_v2_v6_R[str(time_point)]._data[contra_right_ind, :], axis = 0))  + np.mean(st_v2_v6_L[str(time_point)]._data[contra_left_ind, :], axis = 0))/2
    ipsi = (np.mean(st_v2_v6_R[str(time_point)]._data[contra_left_ind, :], axis = 0) + (np.mean(st_v2_v6_L[str(time_point)]._data[contra_right_ind, :], axis = 0)))/2
    li_t_st = (contra-ipsi)/(contra+ipsi)
    #plt.plot(li_t[0:300], color = 'k')
    filtered_data_st = medfilt(li_t_st, kernel_size=21)

# =============================================================================
#     time_point = 'v6'
#    
# 
#     contra =  ((np.mean(st_v2_v6_R[str(time_point)]._data[contra_right_ind, :], axis = 0))  + np.mean(st_v2_v6_L[str(time_point)]._data[contra_left_ind, :], axis = 0))/2
#     ipsi = (np.mean(st_v2_v6_R[str(time_point)]._data[contra_left_ind, :], axis = 0) + (np.mean(st_v2_v6_L[str(time_point)]._data[contra_right_ind, :], axis = 0)))/2
#     li_t_st_v6 = ((contra) - (ipsi))/((contra) + (ipsi))
#     #plt.plot(li_t[0:300], color = 'k')
#     filtered_data_st_v6 = medfilt(li_t_st_v6, kernel_size=51)
# =============================================================================
    
    plt.figure()
    plt.plot((np.mean(st_v2_v6_R[str(time_point)]._data[contra_right_ind, :], axis = 0)) , color = 'r', label = 'contra_right')
    plt.plot( np.mean(st_v2_v6_L[str(time_point)]._data[contra_left_ind, :], axis = 0), color = 'orange', label = 'contra_left')
    plt.plot(np.mean(st_v2_v6_R[str(time_point)]._data[contra_left_ind, :], axis = 0), color = 'g', label = 'ipsi_right')
    plt.plot( np.mean(st_v2_v6_L[str(time_point)]._data[contra_right_ind, :], axis = 0), color = 'b', label = 'ipsi_left')
    plt.plot(contra, color = 'k')
    plt.plot(abs(ipsi), color = 'grey')
    plt.legend()
    plt.show()
    
    
    
    
    contra =  (np.mean(HC_R[0]._data[contra_right_ind, :], axis = 0)  + np.mean(HC_L[0]._data[contra_left_ind, :], axis = 0))/2
    ipsi = (np.mean(HC_R[0]._data[contra_left_ind, :], axis = 0) + np.mean(HC_L[0]._data[contra_right_ind, :], axis = 0))/2
    li_t_hc = (contra-ipsi)/(contra+ipsi)
    filtered_data_hc = medfilt(li_t_hc, kernel_size=21)
    

    
    plt.figure()
    plt.plot(filtered_data_hc[0:250], color = 'k', label = 'HC')
    plt.plot(filtered_data_st[0:250], color = 'r', label = 'Stroke V2')
    #plt.plot(filtered_data_st_v6[0:250], color = 'b', label = 'Stroke V6')
    plt.ylim([-10, 10])
    plt.xlabel('Time (ms)')
    plt.ylabel('Laterality Index')
    #plt.axvspan(100, 130, facecolor='lightskyblue', alpha=0.5, hatch='/')
    plt.axvspan(100, 135, facecolor='pink', alpha=0.5)
    plt.legend()
    plt.show()


        
    fig.savefig(save_folder_peak + 'Laterality Index.svg', overwrite = True)
    
    

def FM_UE_plotting(amp_l_arr_st, amp_r_arr_st, fuma_score_swap, save_folder_peak): 
    from matplotlib.legend_handler import HandlerBase
    fig, ax  = plt.subplots(1, 2, figsize= (16, 4)) 
    
    ax[0].set_xlabel('FM-UE', fontweight ='bold', fontsize = 12)
    ax[0].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 0]
    Y = np.concatenate((amp_l_arr_st[2,:, 0], amp_r_arr_st[2,:, 0]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[0].scatter(X, Y, color=["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"])
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[0].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[0].text(16, 1, f'R = {np.round(r, 3)}, p = {np.round(p, 3)}', fontweight = 'bold', fontsize = 12)
    ax[0].set_xlim(5, 25)
    ax[0].set_ylim(-2, 1.5)
    ax[0].set_title('V2: FM-UE v Amplitude of HC Cluster', fontweight = 'bold', fontsize = 14 )
    
    list_color  = ["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"]
    list_mak    = ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"]
    list_lab    = ['Sub 1','Sub 2','Sub 3', 'Sub 4', 'Sub 5', 'Sub 6', 'Sub 7 ', 'Sub 8', 'Sub 9', 'Sub 10']
    
    ax[0] = plt.gca()
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent, width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="", marker=tup[1],color=tup[0], transform=trans)]
    
    
    fig.legend(list(zip(list_color,list_mak)), list_lab, handler_map={tuple:MarkerHandler()}, bbox_to_anchor=(0.98, 0.8)) 
    
    
    
    ax[1].set_xlabel('FM-UE', fontweight ='bold', fontsize = 12)
    ax[1].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 4]
    Y = np.concatenate((amp_l_arr_st[2,:, 4], amp_r_arr_st[2,:, 4]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[1].scatter(X, Y, color=["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"])
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[1].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[1].text(16, 1, f'R = {np.round(r, 3)}, p = {np.round(p, 3)}', fontweight = 'bold', fontsize = 12)
    ax[1].set_xlim(5, 25)
    ax[1].set_ylim(-2, 1.5)
    ax[1].set_title('V6: FM-UE v Amplitude of HC Cluster', fontweight = 'bold' , fontsize = 14)
    plt.show()
    fig.savefig(save_folder_peak + 'FM_UE_HC.svg')
    
    
    
    
    fig, ax  = plt.subplots(1, 2, figsize= (16, 4)) 
    ax[0].set_xlabel('FM-UE', fontweight ='bold', fontsize = 12)
    ax[0].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 0]
    Y = np.concatenate((amp_l_arr_st[4,:, 0], amp_r_arr_st[4,:, 0]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[0].scatter(X, Y, color=["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"])
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[0].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[0].text(16, 1, f'R = {np.round(r, 3)}, p = {np.round(p, 4)}', fontweight = 'bold', fontsize = 12)
    ax[0].set_xlim(5, 25)
    ax[0].set_ylim(-2, 1.5)
    ax[0].set_title('V2: FM-UE v Amplitude of Sensorimotor Region', fontweight = 'bold' , fontsize = 14)
    
    
    list_color  = ["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"]
    list_mak    = ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"]
    list_lab    = ['Sub 1','Sub 2','Sub 3', 'Sub 4', 'Sub 5', 'Sub 6', 'Sub 7 ', 'Sub 8', 'Sub 9', 'Sub 10']
    
    ax[0] = plt.gca()
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent, width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="", marker=tup[1],color=tup[0], transform=trans)]
    
    
    fig.legend(list(zip(list_color,list_mak)), list_lab, handler_map={tuple:MarkerHandler()}, bbox_to_anchor=(0.98, 0.8)) 
    
    
    
    ax[1].set_xlabel('FM-UE', fontweight ='bold', fontsize = 12)
    ax[1].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 4]
    Y = np.concatenate((amp_l_arr_st[4,:, 4], amp_r_arr_st[4,:, 4]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[1].scatter(X, Y, color=["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"])
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[1].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[1].text(16, 1, f'R = {np.round(r, 3)}, p = {np.round(p, 4)}', fontweight = 'bold', fontsize = 12)
    ax[1].set_xlim(5, 25)
    ax[1].set_ylim(-2, 1.5)
    ax[1].set_title('V6: FM-UE v Amplitude of Sensorimotor Region', fontweight = 'bold' , fontsize = 14)
           

    plt.show()
    fig.savefig(save_folder_peak + 'FM_UE_somatosensory.svg')
    
    
    
   
    
    
def FM_UE_plotting_hand(amp_l_arr_st, amp_r_arr_st, fuma_score_swap, save_folder_peak): 
    from matplotlib.legend_handler import HandlerBase
    fig, ax  = plt.subplots(1, 2, figsize= (16, 4)) 
    
    ax[0].set_xlabel('FM-UE', fontweight ='bold', fontsize = 12)
    ax[0].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 0]
    Y = np.concatenate((amp_l_arr_st[2,:, 0], amp_r_arr_st[2,:, 0]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[0].scatter(X, Y, color=["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"])
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[0].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[0].text(3, 2, f'R = {np.round(r, 3)}, p = {np.round(p, 2)}', fontweight = 'bold', fontsize = 12)
    ax[0].set_xlim(-1, 7)
    ax[0].set_ylim(-2, 3)
    ax[0].set_title('V2: FM-UE v Amplitude of HC Cluster', fontweight = 'bold', fontsize = 14 )
    
    list_color  = ["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"]
    list_mak    = ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"]
    list_lab    = ['Sub 1','Sub 2','Sub 3', 'Sub 4', 'Sub 5', 'Sub 6', 'Sub 7 ', 'Sub 8', 'Sub 9', 'Sub 10']
    
    ax[0] = plt.gca()
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent, width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="", marker=tup[1],color=tup[0], transform=trans)]
    
    
    fig.legend(list(zip(list_color,list_mak)), list_lab, handler_map={tuple:MarkerHandler()}, bbox_to_anchor=(0.98, 0.8)) 
    
    
    
    ax[1].set_xlabel('FM-UE', fontweight ='bold', fontsize = 12)
    ax[1].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 4]
    Y = np.concatenate((amp_l_arr_st[2,:, 4], amp_r_arr_st[2,:, 4]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[1].scatter(X, Y, color=["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"])
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[1].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[1].text(3, 2, f'R = {np.round(r, 3)}, p = {np.round(p, 2)}', fontweight = 'bold', fontsize = 12)
    ax[1].set_xlim(-1, 7)
    ax[1].set_ylim(-2, 3)
    ax[1].set_title('V6: FM-UE v Amplitude of HC Cluster', fontweight = 'bold' , fontsize = 14)
    plt.show()
    fig.savefig(save_folder_peak + 'FM_UE_HC_hand.svg')
    
    
    
    
    fig, ax  = plt.subplots(1, 2, figsize= (16, 4)) 
    ax[0].set_xlabel('FM-UE', fontweight ='bold', fontsize = 12)
    ax[0].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 0]
    Y = np.concatenate((amp_l_arr_st[4,:, 0], amp_r_arr_st[4,:, 0]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[0].scatter(X, Y, color=["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"])
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[0].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[0].text(3, 2, f'R = {np.round(r, 3)}, p = {np.round(p, 4)}', fontweight = 'bold', fontsize = 12)
    ax[0].set_xlim(-1, 7)
    ax[0].set_ylim(-2, 3)
    ax[0].set_title('V2: FM-UE v Amplitude of Sensorimotor Region', fontweight = 'bold' , fontsize = 14)
    
    
    list_color  = ["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"]
    list_mak    = ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"]
    list_lab    = ['Sub 1','Sub 2','Sub 3', 'Sub 4', 'Sub 5', 'Sub 6', 'Sub 7 ', 'Sub 8', 'Sub 9', 'Sub 10']
    
    ax[0] = plt.gca()
    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent, width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="", marker=tup[1],color=tup[0], transform=trans)]
    
    
    fig.legend(list(zip(list_color,list_mak)), list_lab, handler_map={tuple:MarkerHandler()}, bbox_to_anchor=(0.98, 0.8)) 
    
    
    
    ax[1].set_xlabel('FM-UE', fontweight ='bold', fontsize = 12)
    ax[1].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 4]
    Y = np.concatenate((amp_l_arr_st[4,:, 4], amp_r_arr_st[4,:, 4]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[1].scatter(X, Y, color=["blue", "orange", "green", "dodgerblue", "purple", "brown", "pink", "gray", "olive", "cyan"])
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[1].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[1].text(3, 2, f'R = {np.round(r, 3)}, p = {np.round(p, 4)}', fontweight = 'bold', fontsize = 12)
    ax[1].set_xlim(-1, 7)
    ax[1].set_ylim(-2, 3)
    ax[1].set_title('V6: FM-UE v Amplitude of Sensorimotor Region', fontweight = 'bold' , fontsize = 14)
           

    plt.show()
    fig.savefig(save_folder_peak + 'FM_UE_somatosensory_hand.svg')
    
    
    
def box_plot_p1_HC_st(lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):

    dict_p1 = {'Control': lat_t_hc[1, :], 'Stroke V2': np.concatenate((lat_r_arr_st[1, :, 0], lat_l_arr_st[1, :, 0]))}
    fig, ax = plt.subplots(figsize= (4, 4.5))
    data_std = np.zeros([1, 2])
    data_mean =  np.zeros([1, 2])
    data_std = [np.std(dict_p1[str(j)]) for i,j in enumerate(list(dict_p1.keys()))]
    data_mean = [np.mean(dict_p1[str(j)]) for i,j in enumerate(list(dict_p1.keys()))]
    
    
    ax.bar( x = np.arange(len(dict_p1.keys())), height = data_mean, yerr= data_std, capsize=4, color ='grey', alpha = 0.7)
    ax.set_xticklabels(['','Control','' ,'Stroke V2']) 
    ax.set_ylabel('Latency (ms)')
    ax.set_title('P1 Latency', fontweight="bold")
    ax.set_ylim([0, 60])
    ax.set_xlim([-0.5, 1.5])
    t_ind , p_ind = stats.ttest_ind(dict_p1[str('Control')], dict_p1[str('Stroke V2')])
    ax.text(0.1, 50, f'P-value = {np.round(p_ind, 3)}', fontweight = 'bold')
    plt.show()
    fig.savefig(save_folder_peak  + 'p1_latency_barplot'+ '.svg', overwrite = True)
    return(p_ind)





def box_plot_p1_all_time_points_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):

    dict_p1 = {'Control': lat_t_hc[component_number, :], 'Stroke V2': np.concatenate((lat_r_arr_st[component_number, :, 0], lat_l_arr_st[component_number, :, 0])),
                                                         'Stroke V3': np.concatenate((lat_r_arr_st[component_number, :, 1], lat_l_arr_st[component_number, :, 1])),
                                                         'Stroke V4': np.concatenate((lat_r_arr_st[component_number, :, 2], lat_l_arr_st[component_number, :, 2])),
                                                         'Stroke V5': np.concatenate((lat_r_arr_st[component_number, :, 3], lat_l_arr_st[component_number, :, 3])), 
                                                         'Stroke V6': np.concatenate((lat_r_arr_st[component_number, :, 4], lat_l_arr_st[component_number, :, 4]))}
  
    
    
    
    data_std = np.zeros([1, 6])
    data_mean =  np.zeros([1, 6])
    data_std = [np.std(dict_p1[str(j)]) for i,j in enumerate(list(dict_p1.keys()))]
    data_mean = [np.mean(dict_p1[str(j)]) for i,j in enumerate(list(dict_p1.keys()))]
    fig, ax = plt.subplots()
    ax.plot(np.array([1, 2, 3, 4, 5]), data_mean[1:], color = 'k')
    ax.bar( x = np.arange(len(dict_p1.keys())), height = data_mean, yerr= data_std, capsize=4, color ='grey', alpha = 0.7)
    ax.set_xticklabels(['Control','Control', 'Stroke V2', 'Stroke V3', 'Stroke V4', 'Stroke V5', 'Stroke V6']) 
    ax.set_ylabel('Latency (ms)')
    ax.set_title('P1 Latency', fontweight="bold")
    t_ind , p_ind = stats.ttest_ind(dict_p1[str('Control')], dict_p1[str('Stroke V2')])
    t_paired, p_paired = stats.ttest_rel(dict_p1[str('Stroke V2')], dict_p1[str('Stroke V5')])
    ax.text(-.1, 65, f'P-value = {np.round(p_ind, 3)}', fontweight = 'bold')
    ax.scatter(.55, 58, marker = '*', color = 'k')
    ax.plot([0, 1], [55, 55], color = 'k')
    ax.plot([0, 0, 0, 0], [55, 54, 53, 52], color = 'k')
    ax.plot([1, 1, 1, 1], [55, 54, 53, 52], color = 'k')
    ax.set_ylim([0, 70])
    plt.show()
    fig.savefig(save_folder_peak  + 'p1_latency_barplot'+ '.svg', overwrite = True)

    return(p_ind)





def N3_cluster_bilaterality(t_r, t_l, pvals_all_r, pvals_all_l, mask_r, mask_l, pos, t_r_hc, t_l_hc, pvals_all_r_hc, pvals_all_l_hc, mask_r_hc, mask_l_hc, save_folder_peak):

   
    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=5)
    fig, sps = plt.subplots(nrows=2, ncols=6, figsize=(20,8))
    plt.style.use('default')
    time_points = ['v2', 'v3', 'v4', 'v5', 'v6']
    time_points_capital = ['V2', 'V3', 'V4', 'V5', 'V6']
    
    for iplot in range(5):
        for ipeak, time_point in enumerate(time_points):
            imask = mask_r[str(time_point)][:,2]
            im = topoplot_2d(channel_names(), t_r[str(time_point)][:,2], pos,
                                clim=[-5,5], axes=sps[0,ipeak], mask=imask, maskparam=maskparam)
            sps[0,ipeak].set_title(f'{time_points_capital[ipeak]}', fontweight = 'bold', fontsize = 20)
            #sps[0, ipeak].text(-0.05, -0.2 , f'p = {np.round(pvals_all_r[str(time_point)][0], 3)}', fontweight = 'bold', fontsize = 14)
            sps[0, ipeak].text(-0.05, -0.2 , f't = {np.round( sum(t_r[str(time_point)][np.where(mask_r[str(time_point)][:, 2] == 1)[0]  , 2]),2)}', fontweight = 'bold', fontsize = 14)
            
            
    for iplot in range(5):
        for ipeak, time_point in enumerate(time_points):
            imask = mask_l[str(time_point)][:,2]
            im = topoplot_2d(channel_names(), t_l[str(time_point)][:,2], pos,
                                clim=[-5,5], axes=sps[1,ipeak], mask=imask, maskparam=maskparam)       
            sps[1,ipeak].set_title(f'{time_points_capital[ipeak]}', fontweight = 'bold', fontsize = 20)
            #sps[1, ipeak].text(-0.05, -0.2 , f'p = {np.round(pvals_all_l[str(time_point)][0], 3)}', fontweight = 'bold', fontsize = 14)
            sps[1, ipeak].text(-0.05, -0.2 , f't = {np.round(sum(t_l[str(time_point)][np.where(mask_l[str(time_point)][:, 2] == 1)[0]  , 2]), 2)}', fontweight = 'bold', fontsize = 14)
    
    
    
    topoplot_2d(channel_names(), t_r_hc[:, 2], pos, clim=[-5,5], axes=sps[0,5], mask=mask_r_hc[:, 2], maskparam=maskparam) 
    sps[0,5].set_title('HC', fontweight = 'bold', fontsize = 20)
    #sps[0,5].text(-0.05, -0.2 , f'p = {np.round(pvals_all_r_hc[0], 3)}', fontweight = 'bold', fontsize = 14)
    sps[0,5].text(-0.05, -0.2 , f't = {np.round(sum(t_r_hc[np.where(mask_r_hc[:, 2] ==1)[0], 2]), 2)}', fontweight = 'bold', fontsize = 14)
    
    topoplot_2d(channel_names(), t_l_hc[:, 2], pos, clim=[-5,5], axes=sps[1,5], mask=mask_l_hc[:, 2], maskparam=maskparam) 
    sps[1,5].set_title('HC', fontweight = 'bold', fontsize = 20)
    #sps[1,5].text(-0.05, -0.2 , f'p = {np.round(pvals_all_l_hc[0], 3)}', fontweight = 'bold', fontsize = 14)
    sps[1,5].text(-0.05, -0.2 , f't = {np.round(sum(t_l_hc[np.where(mask_l_hc[:, 2] ==1)[0], 2]), 2)}', fontweight = 'bold', fontsize = 14)
    
    
    
    
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad=0.04)
    cb.set_label('t-value', rotation = 90, fontsize = 12)
    plt.show()
    fig.savefig(save_folder_peak + 'N3_cluster_bilaterality' + '.svg') 
    
    
    
    
def FM_UE_L_R_all(fuma_score_swap, save_folder_peak):
    time_points = ['v2', 'v3', 'v4', 'v5', 'v6']
    fig, ax  = plt.subplots(1, 3, figsize= (14, 5))  
    ax[0].errorbar(range(len(time_points)),  np.mean(fuma_score_swap[0:6, :], axis =0 ), np.std( fuma_score_swap[0:6, :], axis = 0), color = 'k', linestyle='None', marker='o')
    ax[0].plot(np.mean(fuma_score_swap[0:6, :], axis =0 ), alpha =0.5, color = 'k')
    ax[0].set_xticklabels(['','V2', 'V3',  'V4',  'V5',  'V6'], fontsize=12)
    ax[0].set_xlabel('Time-points', fontsize=12, fontweight = 'bold')
    ax[0].set_ylabel('FM-UE score', fontsize=12, fontweight = 'bold')
    ax[0].set_title('Left Paretic Hand')
    ax[0].set_ylim([5, 21])
    
    ax[1].errorbar(range(len(time_points)),  np.mean(fuma_score_swap[6:, :], axis =0 ), np.std(fuma_score_swap[6:, :], axis = 0), color = 'k' , linestyle='None', marker='o')
    ax[1].plot(np.mean(fuma_score_swap[6:, :], axis =0 ), alpha =0.5, color = 'k')
    ax[1].set_xticklabels( ['','V2', 'V3',  'V4',  'V5',  'V6'], fontsize=12)
    ax[1].set_xlabel('Time-points', fontsize=12, fontweight = 'bold')
    ax[1].set_ylabel('FM-UE score', fontsize=12, fontweight = 'bold')
    ax[1].set_title('Right Paretic Hand')
    ax[1].set_ylim([5, 21])
    
    ax[2].errorbar(range(len(time_points)),  np.mean(fuma_score_swap, axis =0 ), np.std(fuma_score_swap, axis = 0), color = 'k', linestyle='None', marker='o')
    ax[2].plot(np.mean(fuma_score_swap, axis =0 ), alpha =0.5, color = 'k')
    ax[2].set_xticklabels( ['','V2', 'V3',  'V4',  'V5',  'V6'], fontsize=12)
    ax[2].set_xlabel('Time-points', fontsize=12, fontweight = 'bold')
    ax[2].set_ylabel('FM-UE score', fontsize=12, fontweight = 'bold')
    ax[2].set_title('All Patients')
    ax[2].set_ylim([5, 21])
    fig.savefig(save_folder_peak  + 'FM_UE_L_R_all'+ '.svg', overwrite = True)
    
    #plt.legend()
    plt.show()
    
    
    
    
def LME_mean_intensity_FuMe(i_time_windows, amp_late, name_time_windows, sub_names_L, sub_names_R, amp_l_arr_st, amp_r_arr_st, save_folder_lme):
    all_res = []         
    for _, side in enumerate(['L', 'R']):
        for i_time_point,time_point in enumerate(['v2', 'v3', 'v4', 'v5', 'v6']):
            if side == 'L':
                res = pd.DataFrame((sub_names_L[str(time_point)], amp_l_arr_st[i_time_windows, :, i_time_point], list(np.repeat(time_point, len(sub_names_L[str(time_point)]))), list(np.repeat('L', len(sub_names_L[str(time_point)])))    )).T
                all_res.append(res)
            elif side == 'R': 
                 res = pd.DataFrame((sub_names_R[str(time_point)],  amp_r_arr_st[i_time_windows, :, i_time_point], list(np.repeat(time_point, len(sub_names_R[str(time_point)]))), list(np.repeat('R', len(sub_names_R[str(time_point)])))    )).T
                 all_res.append(res)
            df_res = pd.concat(all_res)    
    df_res.rename(columns={0: 'sub', 1:'data', 2:'visit', 3:'stim_side'})            
    df_res.to_excel(save_folder_lme + f'{name_time_windows}_' + f'{amp_late}.xlsx', sheet_name = f'{name_time_windows}')     
    
    
    
    
    
    
    
def contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component, text):    
    data_contra_s = np.zeros([10, end_time-1000])
    evokeds_all_L_corrected = evokeds_all_L[str('v2')]
    for i1,i in enumerate(evokeds_all_R[str('v2')]):
        data_contra_s[i1+6, :] = np.mean(evokeds_all_R[str('v2')][i1].data[contra_right_ind,  1000:end_time], axis = 0)    
    for i1,i in enumerate(evokeds_all_L[str('v2')]):
        data_contra_s[i1, :] = np.mean(evokeds_all_L_corrected[i1].data[contra_left_ind,  1000:end_time], axis = 0)
        
    data_ipsi_s = np.zeros([10, end_time-1000])
    for i1,i in enumerate(evokeds_all_R[str('v2')]):
        data_ipsi_s[i1+6, :] = np.mean(evokeds_all_R[str('v2')][i1].data[contra_left_ind,  1000:end_time], axis = 0)    
    for i1,i in enumerate(evokeds_all_L[str('v2')]):
        data_ipsi_s[i1, :] = np.mean(evokeds_all_L_corrected[i1].data[contra_right_ind,  1000:end_time], axis = 0)
        
        
           
        
    data_contra_c = np.zeros([17, end_time-1000])
    for i1,i in enumerate(evokeds_all_R_hc):
        data_contra_c[i1+11, :] = np.mean(evokeds_all_R_hc[i1].data[contra_right_ind,  1000:end_time], axis = 0)    
    for i1,i in enumerate(evokeds_all_L_hc):
        data_contra_c[i1, :] = np.mean(evokeds_all_L_hc[i1].data[contra_left_ind,  1000:end_time], axis = 0)
        
        
    data_ipsi_c = np.zeros([17, end_time-1000])
    for i1,i in enumerate(evokeds_all_R_hc):
        data_ipsi_c[i1+11, :] = np.mean(evokeds_all_R_hc[i1].data[contra_left_ind,  1000:end_time], axis = 0)    
    for i1,i in enumerate(evokeds_all_L_hc):
        data_ipsi_c[i1, :] = np.mean(evokeds_all_L_hc[i1].data[contra_right_ind,  1000:end_time], axis = 0)
    
    
    data_ipsi_s[1, 0:60]= np.mean(data_ipsi_s[1, 60:65])
    
    data_ipsi_c[14, 0:40]= np.mean(data_ipsi_c[14, 60:65])
    data_ipsi_c[15, 0:60]= np.mean(data_ipsi_c[15, 60:65])
    def confidence_interval(data):
        # Parameters
        confidence_level = 0.95  # 95% confidence interval
        alpha = 1 - confidence_level
        
        # Calculate the mean and standard error for each time point across all time series
        means = np.mean(data, axis=0)
        sems = stats.sem(data, axis=0)
        
        # Calculate the margin of error
        margin_of_error = sems * stats.t.ppf(1 - alpha/2., data.shape[0] - 1)
        
        # Calculate the confidence interval
        lower_bound = means - margin_of_error
        upper_bound = means + margin_of_error
        
        return(lower_bound, upper_bound, means, sems)
    
    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(7, 10))
    time_points = evokeds_all_R[str('v2')][0].times[1000:end_time]
    # stroke
    lower_bound, upper_bound, means, sems = confidence_interval(zscore(data_contra_s, axis  =1))
    ax[0].plot(time_points, means, label='S', color='r')
    ax[0].fill_between(time_points, means - sems, means + sems, color='r', alpha=0.1)
    # control
    lower_bound, upper_bound, means, sems = confidence_interval(zscore(data_contra_c, axis  =1 ))
    ax[0].plot(time_points, means, label='C', color='k')
    ax[0].fill_between(time_points, means - sems, means + sems, color='k', alpha=0.15)
    ax[0].set_xlabel('Latency (ms)')
    ax[0].set_ylim(-2, 2)
    ax[0].set_ylabel('Mean Amplitude (µV)')
    ax[0].set_title('Cotralateral response')
# =============================================================================
#     if component == 'N2':
#         ax[0].vlines(x = [0.015, 0.05], ymin = 0, ymax = 1.5, color = 'grey', linewidth = 1, linestyle = '-.')
#     elif component == 'N1':
#         ax[0].vlines(x = [0.045, 0.080], ymin = 0, ymax = -1.8, color = 'grey', linewidth = 1, linestyle = '-.')
#     elif component == 'P2':
#         ax[0].vlines(x = [0.140, 0.22], ymin = -0.5, ymax = 1, color = 'grey', linewidth = 1, linestyle = '-.')
# =============================================================================
        
    ax[0].legend()
    plt.show()
    
    
    
    # Plotting
    time_points = evokeds_all_R[str('v2')][0].times[1000:end_time]
    # stroke
    lower_bound, upper_bound, means, sems = confidence_interval(zscore(data_ipsi_s, axis = 1))
    ax[1].plot(time_points, means, label='S', color='r')
    ax[1].fill_between(time_points, means - sems, means + sems, color='r', alpha=0.2)
    # control
    lower_bound, upper_bound, means, sems = confidence_interval(zscore(data_ipsi_c, axis =1))
    ax[1].plot(time_points, means, label='C', color='k')
    ax[1].fill_between(time_points, means - sems, means + sems, color='k', alpha=0.2)
    ax[1].set_xlabel('Latency (ms)')
    ax[1].set_ylim(-2, 2)
    ax[1].set_ylabel('Mean Amplitude (µV)')
    ax[1].set_title('Ipsilateral response')
    if component == 'N2':
        ax[1].vlines(x = [0.100, 0.135], ymin = -1, ymax = 1, color = 'b', linewidth = 1)
    ax[1].legend()
    plt.show()
    fig.suptitle(f'{text}')
    fig.savefig(save_folder_peak  + f'{text}'+ 'contra_ipsi_william'+ '.svg', overwrite = True)
    
    
    
    
    
    
    
    
    
    