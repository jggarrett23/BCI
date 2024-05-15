import numpy as np
import math
import os
from scipy.io import loadmat
import time
from datetime import datetime
import ast
import pickle

# EEGNet-specific imports
from arl_eegmodels.EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import mne
from mne.decoding import Scaler
import argparse

import sys


def slidingTime_kf_training(X, y, nEpochs, batch_size, cv, nChans, 
                            kernLength, time, twindow_length, tDelta_length, F1, D):

    
    # create test set of whole epochs (prevents bleeding of training data into test data)
    #test_size = math.ceil(X.shape[0]*0.1)
    #splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    #train_val_idx, test_idx = next(splitter.split(data_scaled, shotLabels))

    # account for imbalance in makes and misses for each subject
    unique_labels, counts = np.unique(y, return_counts=True)
    class_weights = dict(zip(unique_labels, (1./counts) * (len(y)/2.0) ))

    # convert labels to one hot encoding (easier than changing the last layer of EEGNet to sigmoid)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    y = enc.fit_transform(y[:,np.newaxis])

    #X_test = X[test_idx]
    #y_test = y[test_idx, :]
        
    #X = X[train_val_idx,:]
    #y = y[train_val_idx, :]
   
    time_acc = []
    time_auc = []
    time_loss = []

    # loop through each time point
    for iTime in range(0,len(time)-twindow_length,tDelta_length):

        twindow_idx = np.arange(iTime,iTime+twindow_length)
        print(f'Training Classifier for data between {time[twindow_idx[0]]} - {time[twindow_idx[-1]]}ms\n')

        fold = 0
        folds_valLoss = []
        folds_valAcc = []
        folds_valAuc = []
        
        for train_idx, val_idx in cv.split(X, y.argmax(axis=-1)):

            # split training and validation data
            X_train = np.take(X[train_idx, :], twindow_idx, axis=2)
            X_val = np.take(X[val_idx, :], twindow_idx, axis=2)

            y_train, y_val = y[train_idx, :], y[val_idx, :]
            
            # clear model to avoid consuming memory since creating new model with each new fold
            K.clear_session()
            
            # configure the EEGNet-8,2,16 model with kernel length of 1/2 sampling rate
            model = EEGNet(nb_classes = len(np.unique(y.argmax(axis=-1))), Chans = nChans, Samples = X_train.shape[2], 
                           dropoutRate = 0.5, kernLength = kernLength, F1 = F1, D = D, F2 = F1*D, 
                           dropoutType = 'Dropout')

            # compile the model and set the optimizers
            model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics = ['accuracy'])


            # set a valid path for your system to record model checkpoints
            #checkpointer = ModelCheckpoint(filepath= os.path.join(modelDir, f'checkpoints/sj{sjNum:02d}_fold{fold+1}_checkpoint.h5'), verbose=1,
                                           #save_best_only=True)

            # Fit model
            history = model.fit(X_train, y_train, batch_size = batch_size, epochs = nEpochs, 
                                    verbose = 0, validation_data=(X_val, y_val),
                                    class_weight = class_weights)

            # Training Metrics
            train_acc = history.history['accuracy'][0]
            #val_acc = history.history['val_accuracy'][0] # unbalanced
            train_loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]
            
            val_preds = model.predict(X_val).argmax(axis=1)
            val_true = y_val.argmax(axis=1)
            
            val_auc = roc_auc_score(val_true, val_preds)
            val_acc = balanced_accuracy_score(val_true, val_preds)
            
            fold += 1

            print(f'\nFold {fold}/{kf.n_splits} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_acc:.2%} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}\n')
            
            
            # check if this model performed better than others
            if folds_valLoss and val_loss < min(folds_valLoss):
                best_model = model
                print(f'Updated Best Model. Validation loss decreased from {min(folds_valLoss):.3f} to {val_loss:.3f}.\n')
            elif not folds_valLoss:
                best_model = model
                

            folds_valLoss.append(val_loss)
            folds_valAcc.append(val_acc)
            folds_valAuc.append(val_auc)
            
            
        folds_valLoss = np.asarray(folds_valLoss)
        folds_valAcc = np.asarray(folds_valAcc)
        folds_valAuc = np.asarray(folds_valAuc)
        
        # Test data metrics
        #test_loss, test_acc = best_model.evaluate(X_test[:,:,twindow_idx,:], y_test)
        bestModel_valAcc = folds_valAcc[folds_valLoss.argmin()]
        print(f'\nBest Model: {bestModel_valAcc:.2%}\n')
        
        time_loss.append(folds_valLoss[np.newaxis,:].T)
        time_acc.append(folds_valAcc[np.newaxis,:].T)
        time_auc.append(folds_valAuc[np.newaxis,:].T)
        
        
        #cv_results['Test Acc'] = test_acc*100

        #print(f'\nTest Acc: {test_acc:.2%}\n')

    time_loss = np.hstack(time_loss)
    time_acc = np.hstack(time_acc)
    time_auc = np.hstack(time_auc)
    
    time_cvResults = {}
    time_cvResults['Folds'] = range(kf.n_splits)
    time_cvResults['Time Loss'] = time_loss
    time_cvResults['Time Acc'] = time_acc * 100
    time_cvResults['Time Auc'] = time_auc
    
    return(time_cvResults)

    
    
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--sjNum", type=int, help='Subject Number')
    parser.add_argument("--nEpochs", type=int, default=10, help="Number of Epochs for training, default=10")
    parser.add_argument("--nFolds", type=int, default=3, help="Number of Folds for cv, default=3")
    parser.add_argument("--F1", type=int, default=8, help='Number of Temporal Filters')
    parser.add_argument("--D", type=int, default=2, help='Number of Spatial Filters')
    parser.add_argument("--tWindow", type=int, default=1000, help='Duration of data crops (ms)')
    parser.add_argument("--tDelta", type=int, default=500, help='Size of each time step (ms)')
    
    args = parser.parse_args()
    
    ##################### Set up directories ######################

    parentDir = '/home/bullock/Gimmick_Ball/Jordan'
    dataDir = os.path.join(parentDir, 'Data')
    modelDir = os.path.join(parentDir, 'Models')
    saveDir = os.path.join(parentDir, 'Classifier_Results')
    
    sjNum = args.sjNum
    
    ############## Load EEG data and behavioral labels ############

    # data should reflect trials that have already gone through epoching/cleaning
    dataDict = loadmat(os.path.join(dataDir, f'sj{sjNum:02d}_EEG_ICAcleaned.mat'))
    EEG = dataDict['EEG']
    allShotsEEG = dataDict['allShotsEEG']

    EEG_fieldnames = EEG.dtype.names
    allShotsEEG_fieldnames = allShotsEEG.dtype.names

    EEG_dataIdx = EEG_fieldnames.index('data')
    shotOutcomeIdx = allShotsEEG_fieldnames.index('scored')
    condIdx = allShotsEEG_fieldnames.index('cond')

    # unwrap so dont have to keep indexing
    EEG = EEG[0][0]

    data = EEG[EEG_dataIdx]

    # change dimensions to be trials x electrodes x time
    data = np.moveaxis(data, [0, 1, 2], [1, 2, 0])

    nTrials = allShotsEEG[0].shape[0]

    shotLabels = [ allShotsEEG[0][iTrial][shotOutcomeIdx].item() for iTrial in range(nTrials) ]

    condLabels = [ allShotsEEG[0][iTrial][condIdx].item() for iTrial in range(nTrials) ]

    ######## Create mne object if want to use built in decoding #######

    srate = EEG[EEG_fieldnames.index('srate')].item()

    time = EEG[EEG_fieldnames.index('times')].squeeze()
    
    nChans = EEG[EEG_fieldnames.index('nbchan')].item()

    ch_names = [ EEG[EEG_fieldnames.index('chanlocs')][0][iChan][0].item() for iChan in range(nChans)]

    EEG_mneInfo = mne.create_info(ch_names, srate, ch_types='eeg')

    EEG_mneInfo.set_montage('standard_1005')

    # scale by 1000 due to scaling sensitivity in deep learning
    data_scaled = data*1000

    # set kernel length to 1/2 of the sampling rate. Should be an integer
    kernLength = math.floor(srate*0.5)
    
    ##################### Apply EEGNet ######################
    
    # reshape data for convolutional layers
    X = data_scaled[:,:,:,np.newaxis]
    y = np.asarray(shotLabels, dtype=np.int64) # make sure labels are integers

    twindow_ms = args.tWindow # define time window size for sliding estimator
    twindow_length = len(np.arange(np.abs(time - 0).argmin(),
                               np.abs(time-twindow_ms).argmin())
                        )
    
    tDelta = args.tDelta # size of each time step
    tDelta_length = len(np.arange(np.abs(time - 0).argmin(),
			       np.abs(time-tDelta).argmin())
			)


    kf = StratifiedKFold(n_splits=args.nFolds, shuffle=True, random_state=0)
    
    F1, D = args.F1, args.D
    nEpochs = args.nEpochs
    batch_size = 64
    
    # Cross validation training
    time_cvResults = slidingTime_kf_training(X, y, nEpochs, batch_size, kf, 
                                                   nChans, kernLength, time, twindow_length, tDelta_length, F1, D)
    
    saveFile_name = os.path.join(saveDir, f'sj{sjNum:02d}_EEGNet_{F1}_{D}_{twindow_ms}_{tDelta}_results.pickle')
    
    with open(saveFile_name, 'wb') as handle:
        pickle.dump(time_cvResults, handle)
