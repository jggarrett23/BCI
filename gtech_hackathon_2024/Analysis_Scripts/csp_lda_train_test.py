import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import load_and_epoch, epoch_slidingWindow_DA, balance_split
from utils import mininum_entropy_combination as MEC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
from scipy.signal import butter, filtfilt
import argparse
import warnings
from tqdm import tqdm

class Custom_CSP(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_components, freq_bands, **kwargs):
        
        self.n_components = n_components
        self.norm_trace = kwargs.get('norm_trace', False)
        self.log = kwargs.get('log', True)
        self.freq_bands = freq_bands
        self.nFreqs = len(freq_bands)
        self.cov_est = kwargs.get('cov_est', 'concat')
    
    def fit(self, X, y):
        
        # N trials must be the first dimension when inputted into cross validate or pipeline
        if X.shape[0] != len(y):
            raise ('Epochs must be first dimension of input.')
        elif X.shape[1] != len(self.freq_bands):
            raise('N Freqs must be second dimension of input.')
        
        csp = CSP(n_components = self.n_components,
                  norm_trace = self.norm_trace,
                  log = self.log,
                  cov_est = self.cov_est
                 )
        
        self.freq_csps = []
        for i in range(self.nFreqs):
            fitted_csp = csp.fit(X[:,i,:], y)
            self.freq_csps.append(fitted_csp)
            
        return self
    
    def transform(self, X):
        
        transformed_Xs = []
        for i in range(self.nFreqs):
            Xnew = self.freq_csps[i].transform(X[:,i,:])
            transformed_Xs.append(Xnew)
        
        allCSP_Xnew = np.hstack(transformed_Xs)
        
        return allCSP_Xnew
    
    def fit_transform(self, X, y, **fit_params):
        return super().fit_transform(X, y=y, **fit_params)

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

    X_test, y_test = con_data[test_condition]
    nFb_bands = X_test.shape[0]

    # Trials must be first dimension for CSP and data augmentation
    # Augment training data using sliding windows
    X_train_DA = []
    y_train = []
    for i in range(nFb_bands):
        fb_X_train, y_train = epoch_slidingWindow_DA(X_train[i, :], train_t, labels=train_labels, 
                                      win_duration=tWin_dur,
                                      win_delta=args.tWindow_overlap)
        print(fb_X_train.shape)
        # improve SNR with minimum entropy combination
        fb_X_train = MEC(fb_X_train, stim_freqs, win_size, srate, nHarmonics=6)
        X_train_DA.append(fb_X_train[np.newaxis, :])

    X_train = np.concatenate(X_train_DA, axis=0).swapaxes(0,1)

    fb_csp = Custom_CSP(n_components=nFreqs, freq_bands=np.arange(nFb_bands), 
                             norm_trace=False, log=True, cov_est='epoch')
    clf = make_pipeline(fb_csp, LDA())
    
    clf.fit(X_train, y_train)
    
    # test model on left out data over time
    print('Testing model....')

    # put trials first
    X_test = X_test.swapaxes(0,1)

    test_time_acc = np.zeros(len(t)-win_size)
    for iTime in tqdm(range(len(t)-win_size)):
        
        x_t = X_test[:, :, :, iTime:iTime+win_size]

        for iBand in range(nFb_bands):
            x_t[iBand, :] = MEC(x_t, stim_freqs, win_size, srate, nHarmonics=6)
        
        acc = clf.score(x_t, y_test)
        test_time_acc[iTime] = acc  
            

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
    plt.savefig(f'subject_{sjNum}_csp_nHarmons_{nHarmonics}_training_{training_condition}_testing{test_condition}_tWin_{tWin_dur}_testAcc.jpg',
                bbox_inches='tight')
    plt.show()