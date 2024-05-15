import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import load_and_epoch, epoch_slidingWindow_DA, balance_split
from pyriemann.estimation import BlockCovariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM
from scipy.signal import butter, filtfilt
import argparse
import warnings
from tqdm import tqdm

# Author: Jordan Garrett
# UCSB Attention Lab

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sjNum', type=int, default=1)
    parser.add_argument('--training_condition', type=int, default=1)
    parser.add_argument('--tWindow_duration', type=float, default=3)
    parser.add_argument('--tWindow_overlap', type=float, default=0.5)
    parser.add_argument('--nHarmonics', type=int, nargs='+', default=3)

    args = parser.parse_args()
    
    rootDir = 'D:/gtech_hackathon_2024'
    dataDir = os.path.join(rootDir, 'SSVEP_Data')
    sjNum = args.sjNum
    training_condition = args.training_condition
    test_condition = int(np.setdiff1d(np.arange(2), training_condition))
    tWin_dur = args.tWindow_duration
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
        harmonics = np.asarray(args.nHarmonics, dtype=int)
        if len(harmonics) == 1:
            nHarmonics = harmonics[0]
            harmonics = np.arange(nHarmonics)+1
        else:
            nHarmonics = len(args.nHarmonics)
            
        nFreqs = len(stim_freqs)*nHarmonics
        data_filt = np.zeros((nFreqs, nTrials, nChans, nTimepoints))
    
        print('Filtering data...')
        cnt = 0
        for iBand, band_c in enumerate(stim_freqs):
            for iHarmonic, h in enumerate(harmonics):
                c = band_c * h
                b, a = butter(5, Wn=(c-0.1, c+0.1), fs=srate, btype='bandpass')
                data_filt[cnt, :] = filtfilt(b, a, data)
                cnt += 1
    
        data_filt = data_filt.reshape(nTrials, nChans*nFreqs, nTimepoints)
        epoch_labels = epoch.events[:, 2]

        con_data.append((data_filt, epoch_labels))
    
    win_size = len(np.arange(0, tWin_dur, 1/srate))
    
    # truncate data to keep only timepoints when stimulation occured
    train_toi = np.arange(abs(t-0).argmin(),abs(t-7).argmin())
    train_t = t[train_toi]

    train_dataset, train_labels = con_data[training_condition]
    X_train = train_dataset[:, :, train_toi]

    # Augment training data using sliding windows
    X_train, y_train = epoch_slidingWindow_DA(X_train, train_t, labels=train_labels, 
                                  win_duration=tWin_dur,
                                  win_delta=args.tWindow_overlap)
    
    X_test, y_test = con_data[test_condition]

    print('Training model...\n')
    cov_estimator = BlockCovariances(estimator='lwf', block_size=int(nChans*nHarmonics))
    train_cov_ests = cov_estimator.transform(X_train)
    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        mdm.fit(train_cov_ests, y_train)

        # test model on left out data over time
        print('Testing model....')
        test_time_acc = np.zeros(len(t)-win_size)
        for iTime in tqdm(range(len(t)-win_size)):
            X_test_t = X_test[:, :, iTime:iTime+win_size]
            test_cov_ests = cov_estimator.transform(X_test_t)
            test_acc = mdm.score(test_cov_ests, y_test)
            test_time_acc[iTime] = test_acc

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
    plt.savefig(f'subject_{sjNum}_fvep_led_nHarmons_{nHarmonics}_training_{training_condition}_testing{test_condition}_tWin_{tWin_dur}_testAcc.jpg',
                bbox_inches='tight')
    plt.show()

    
    
    
