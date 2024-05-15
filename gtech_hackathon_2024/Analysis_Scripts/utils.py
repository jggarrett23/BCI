import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
# Author: Jordan Garrett
# UCSB Attention Lab

def load_and_epoch(raw_file, tmin=-2.5, tmax=7.5):

    """
    Load and epoch eeg data.

    Parameters
    ----------
    raw_file : str, 
        file path for single individual raw .EDF file
    """
    
    raw = mne.io.read_raw_edf(raw_file, verbose=0)
    events = mne.find_events(raw, stim_channel = '10')
    # based on classInfo_4_5.m
    y = np.tile(np.arange(4), 5)
    events[:, 2] = y

    mapping = {0:'9', 1:'10', 2:'12', 3:'15'}

    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )
    raw.set_annotations = annot_from_events

    epoch = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, verbose=0)

    return epoch

def epoch_slidingWindow_DA(data, time, labels=None, win_duration=3, 
                     win_delta=0.5, srate=256., chan_picks=None):

    """
    Split epoched eeg data into crops of overlapping windows.

    Parameters
    ----------
    data : mne.epochs.Epochs object or numpy.ndarray, shape (epochs x channels x time)
        eeg data
    time : numpy.ndarray
        array of time points for each epoch
    labels : None or iterable, default=None
        corresponding event for each epoch (e.g., SSVEP flicker). If numpy.ndarray used as input eeg data, then labels must be provided
    win_duration : float, default = 3 
        duration of each crop in seconds.
    win_delta : float, default = 0.5
        percentage of overlap between window crops
    srate : float, default = 256.
        sampling rate of the eeg data
    chan_picks : iterable, default = None
        channel indices that want to keep.
    """

    if type(data) == mne.epochs.Epochs:
        labels = data.events[:, 2]
        
        if chan_picks is None:
            data = data.get_data()
        else:
            data = data.get_data()[:, chan_picks, :]
            
    elif type(data) == np.ndarray:
        if data.ndim != 3:
            raise('Data should be in format epochs x channels x time')
        elif labels is None:
            raise('Flicker frequency labels must be given for each epoch')
    else:
        raise('Data should either be an mne Epochs object or numpy array')
    
    # data should be in epochs x channels x time
    nEpochs, nChans, nTimepoints = data.shape
    
    win_size = len(np.arange(0, win_duration, 1/srate))
    delta = 1-win_delta
    w_overlap = math.floor(win_size * delta) if math.floor(win_size * delta) else 1
    tWindow_startIdxs = np.arange(0, nTimepoints-win_size, w_overlap)
    nWindows = len(tWindow_startIdxs)

    # crop data into sliding windows
    X = np.zeros((nEpochs, nWindows, nChans, win_size))
    y = np.zeros((nEpochs, nWindows))
    for iEpoch in range(nEpochs):
      for iCrop, w_start in enumerate(tWindow_startIdxs):
        w_end = w_start + win_size
        X[iEpoch,iCrop, :] = data[iEpoch, :, w_start:w_end]
        y[iEpoch,iCrop] = labels[iEpoch]

    X = X.reshape(-1, nChans, win_size)
    y = y.flatten()
    
    return X, y

def balance_split(labels, test_size=0.2, random_state=0):

    """
    Split data in fashion that yields equal number of labels per class

    Parameters
    ----------
    labels : iterable
        class labels for each sample
    test_size : float
        percentage of data to keep for testing set
    random_state : int, default=0
        seed for random number generator
    """
    
    np.random.seed(random_state)

    # get frequency of each label
    uniq_labels, counts = np.unique(labels, return_counts=True)

    # determine number of trials for validation
    nTest_trials = math.ceil(min(counts) * test_size)

    counts -= nTest_trials

    min_trials = min(counts) - 1

    # get trial indices for each label
    label_idxs = [np.argwhere(labels == i).reshape(-1) for i in uniq_labels]

    # loop over labels
    train_idxs = []
    test_idxs = []

    for iLabel_idx in label_idxs:
        # randomly permute label trial indices
        shuffIdx = np.random.permutation(len(iLabel_idx))

        # extract validation samples indices
        test_idx = iLabel_idx[shuffIdx[:nTest_trials]]

        # extract training samples indices
        train_idx = iLabel_idx[shuffIdx[nTest_trials:]]

        # truncate training samples indices
        train_idx = train_idx[:min_trials]

        train_idxs.append(train_idx)
        test_idxs.append(test_idx)

    train_idxs = np.hstack(train_idxs)
    test_idxs = np.hstack(test_idxs)

    # return iterable of tuples with (train, test) indices to pass as input into sklearn
    yield train_idxs, test_idxs

def mininum_entropy_combination(data, freqs, nTimepoints, Fs, nHarmonics=6):

    X = []
    t = (np.arange(nTimepoints)+1)/Fs
    for k in range(1, nHarmonics+1):
        # sin and cosine wave of stimulation frequencies
        sub_X = np.zeros((nTimepoints, 2))
        for f in freqs:
            sub_X[:, 0] = np.sin(2*np.pi*f*k*t)
            sub_X[:, 1] = np.cos(2*np.pi*f*k*t)
            X.append(sub_X)
            
    X = np.hstack(X)

    # Compute nuisance signals in data
    # Y_nuisance = Y - X*pinv(X.T*X)*X.T*Y

    # check dimensions of data
    # channels should be last dimension
    if data.ndim == 3:
        n, tt, c = data.shape
    else:
        tt, c = data.shape

    # incase time is last dimension
    if c == nTimepoints:
        if data.ndim == 3:
            data = data.swapaxes(1,-1)
        else:
            data = data.T

    inv_term = X @ np.linalg.pinv(X.T @ X)
    nuisance = data - inv_term @ X.T @ data

    # apply PCA to nuisance matrix and remove nuisance singals from real data
    # reshape to be either epoch x chan x time or chan x time
    pca = PCA()
    data_cleaned = np.zeros((data.shape))
    if data.ndim == 3:
        for i in range(n):
            pca.fit(nuisance[i,:])
            data_cleaned[i, :] = pca.transform(data[i, :])
        data_cleaned = data_cleaned.reshape(n, c, tt)
    else:
        pca.fit(nuisance)
        data_cleaned = pca.transform(data).T

    return data_cleaned



        

    