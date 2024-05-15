import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import load_and_epoch, epoch_slidingWindow_DA, balance_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import BlockCovariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM
from scipy.signal import butter, filtfilt
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
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
    parser.add_argument('--nFolds', type=int, default=3)

    args = parser.parse_args()
    
    rootDir = 'D:/gtech_hackathon_2024'
    dataDir = os.path.join(rootDir, 'SSVEP_Data')
    
    epoch = load_and_epoch(os.path.join(dataDir, f'subject_{args.sjNum}_fvep_led_training_{args.training_condition}.EDF'),
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
    nFolds = args.nFolds
    epoch_labels = epoch.events[:, 2]
    win_size = len(np.arange(0, args.tWindow_duration, 1/srate))
    cv_training_acc = np.zeros((nFolds,))

    # truncate data to keep only timepoints when stimulation occured
    train_toi = np.arange(abs(t-0).argmin(),abs(t-7).argmin())
    train_t = t[train_toi]
    
    cv_val_acc_time = np.zeros((nFolds, len(t)-win_size))
    cov_estimator = BlockCovariances(estimator='lwf', block_size=int(nChans*nHarmonics))
    for iFold in range(nFolds):
        print(f'Fold {iFold+1} / {nFolds}')
        train_idx, test_idx = next(balance_split(epoch_labels, test_size=0.2, random_state=iFold+10))
        X_train = np.take(data_filt[:, :, train_toi], train_idx, axis=0)
        X_test = data_filt[test_idx, :]
        y_train, y_test = epoch_labels[train_idx], epoch_labels[test_idx]

        # Augment training data using sliding windows
        train_crops, y_train = epoch_slidingWindow_DA(X_train, train_t, labels=y_train, 
                                      win_duration=args.tWindow_duration,
                                      win_delta=args.tWindow_overlap)
        
        train_cov_ests = cov_estimator.transform(train_crops)
        mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            print('Fitting model....')
            mdm.fit(train_cov_ests, y_train)

            # test model on left out data over time
            print('Testing model....')
            for iTime in tqdm(range(len(t)-win_size)):
                X_test_t = X_test[:, :, iTime:iTime+win_size]
                test_cov_ests = cov_estimator.transform(X_test_t)
                test_acc = mdm.score(test_cov_ests, y_test)
                cv_val_acc_time[iFold, iTime] = test_acc

    plt.figure(figsize=(10,5))
    plt.plot(t[win_size:], (1-cv_val_acc_time).mean(axis=0)*100)
    #plt.axhline(25., linestyle='--', color='k', alpha=0.25, label='chance')
    plt.axvline(0, color='r', label='Stim. Onset')
    plt.ylabel('Validation Error %')
    plt.xlabel('Time (s)')
    plt.ylim(0, 100)
    plt.legend(frameon=False)
    plt.title(f'Subject: {sjNum} Training: {args.training_condition} \n {args.tWindow_duration} s window')
    plt.grid()
    plt.savefig(f'subject_{args.sjNum}_fvep_led_training_{args.training_condition}_tWin_{args.tWindow_duration}_validationAcc.jpg',
                bbox_inches='tight')
    plt.show()
                                  
    