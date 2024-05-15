import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import load_and_epoch, epoch_slidingWindow_DA, balance_split
from utils import mininum_entropy_combination as MEC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
import argparse
import warnings
from tqdm import tqdm

# Author: Jordan Garrett
# UCSB Attention Lab

# CCA
def gen_cca_ref(freq, n_harmonics, t):

  nTimepoints = len(t)
  ref = np.zeros((2*n_harmonics, nTimepoints))
  harmon_cnt = 1
  for i in range(0, ref.shape[0], 2):
    ref[i, :] = np.sin(2*np.pi*freq*harmon_cnt*t)
    ref[i+1, :] = np.cos(2*np.pi*freq*harmon_cnt*t)
    harmon_cnt += 1

  return ref

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sjNum', type=int, default=1)
    parser.add_argument('--training_condition', type=int, default=1)
    parser.add_argument('--tWindow_duration', type=float, default=3)
    parser.add_argument('--tWindow_overlap', type=float, default=0.5)
    parser.add_argument('--nHarmonics', type=int, default=6)
    parser.add_argument('--filter_bank', type=bool, default=False)
    parser.add_argument('--fb_method', type=str, default='harmonic_bands')

    args = parser.parse_args()
    
    rootDir = 'D:/gtech_hackathon_2024'
    dataDir = os.path.join(rootDir, 'SSVEP_Data')
    sjNum = args.sjNum
    training_condition = args.training_condition
    test_condition = int(np.setdiff1d(np.arange(2), training_condition))
    tWin_dur = args.tWindow_duration
    fb = args.filter_bank
    con_data = []
    for iSession in range(1,3):

        print(f'Loading and epoching session {iSession} data...\n')
        epoch = load_and_epoch(os.path.join(dataDir, f'subject_{sjNum}_fvep_led_training_{iSession}.EDF'),
                              tmin=-2.5, tmax=7.5)
        
        
        srate = epoch.info['sfreq']
        
        t = np.arange(-3, 7, 1/srate)
    
        data = epoch.get_data()[:, 2:9, :]
        nTrials, nChans, nTimepoints = data.shape
        
        # filter data
        stim_freqs = np.asarray([9, 10, 12, 15])
        nHarmonics = args.nHarmonics
            
        nFreqs = len(stim_freqs)
        if fb:
            fb_method = args.fb_method
            if fb_method == 'even_bins':
                fb_band_delta = 8
                fb_band_starts = np.arange(8, 88+fb_band_delta, fb_band_delta)
                fb_band_lims = [(i, i+fb_band_delta) for i in fb_band_starts]
                fb_a, fb_b = 1.75, 0.25
            elif fb_method == 'harmonic_bands':
                fb_band_lims = []
                nBands = 2
                fb_a, fb_b = 1.25, 0.25
                for b in range(1, nBands+1):
                    fb_band_lims.append((b*8, min(b*16, 88)))
                    
            nFb_bands = len(fb_band_lims)
            
            data_filt = np.zeros((nFb_bands, nTrials, nChans, nTimepoints))
    
            print('Filtering data...')
            for iBand, band_lims in enumerate(fb_band_lims):
                b_start, b_end = band_lims
                b, a = butter(5, Wn=(b_start-2, b_end+2), fs=srate, btype='bandpass')
                data_filt[iBand, :] = filtfilt(b, a, data)
                
            data = data_filt
        else:
            # notch filter (line noise 50 Hz)
            data = mne.filter.notch_filter(data, Fs=srate, freqs=50)
            data = data[np.newaxis,:]

        epoch_labels = epoch.events[:, 2]

        con_data.append((data, epoch_labels))
    
    win_size = len(np.arange(0, tWin_dur, 1/srate))
    
    # truncate data to keep only timepoints when stimulation occured
    train_toi = np.arange(abs(t-0).argmin(),abs(t-7).argmin())
    train_t = t[train_toi]

    train_dataset, train_labels = con_data[training_condition]
    X_train = np.take(train_dataset, train_toi, axis=-1)

    del X_train
    
    #generate reference template for ssvep flicker frequencies and harmonics
    cca_refs = [gen_cca_ref(i, nHarmonics, np.arange(win_size)+1/srate) for i in stim_freqs]
    cca = CCA(n_components=1)

    X_test, y_test = con_data[test_condition]
    nFb_bands = X_test.shape[0]

    # test model on left out data over time
    print('Testing model....')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        cors = np.zeros((nFb_bands, X_test.shape[1], nFreqs, len(t)-win_size))
        for iBand in tqdm(range(nFb_bands)):
            for iTrial in range(X_test.shape[1]):
              for iTime in range(len(t)-win_size):
                  x = X_test[iBand, iTrial, :, iTime:iTime+win_size]

                  # clean data using minimum covariance combination
                  x = MEC(x, stim_freqs, nTimepoints, srate, nHarmonics=nHarmonics)
                  
                  # fit cca
                  for iFlick in range(nFreqs):
                      x_c, y_c = cca.fit_transform(x.T, cca_refs[iFlick].T)
                      rho, _ = pearsonr(x_c.squeeze(), y_c.squeeze())
                      cors[iBand, iTrial, iFlick, iTime] = rho
    if not fb:
        cors = cors.squeeze()
    else:
        w = np.power(np.arange(1, nFb_bands+1), -fb_a) + fb_b
        cors = (w @ cors.swapaxes(0, 2)**2).swapaxes(0,1)
    
    test_preds = cors.argmax(axis=1)
    test_time_acc = (test_preds == y_test[:, np.newaxis].repeat(len(t)-win_size, axis=-1)).mean(axis=0)
    

    plt.figure(figsize=(10,5))
    plt.plot(t[win_size:], (1-test_time_acc)*100)
    plt.axvline(0, color='r', label='Stim. Onset')
    plt.axhline(75., linestyle='--', color='k', alpha=0.25, label='Chance')
    plt.ylabel('Test Error %')
    plt.xlabel('Time (s)')
    plt.ylim(0, 100)
    plt.legend(frameon=False)
    plt.title(f'Subject: {sjNum} Training: {args.training_condition} \n {tWin_dur} s window')
    plt.grid()
    plt.savefig(f'subject_{sjNum}_cca_nHarmons_{nHarmonics}_training_{training_condition}_testing{test_condition}_tWin_{tWin_dur}_testAcc.jpg',
                bbox_inches='tight')
    plt.show()

    
    
    
