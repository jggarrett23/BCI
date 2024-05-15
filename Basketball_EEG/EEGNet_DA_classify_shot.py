import numpy as np
import math
import os
from scipy.io import loadmat
import time
from datetime import datetime
import ast
import pickle


from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import keras_tuner as kt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import mne
from mne.decoding import Scaler
import argparse

import sys

from Custom_EEGNet import Hyper_EEGNet

####### Data Augmentation Methods #####

# sliding window method
def slidingCrops_DA(X, y, trial_idx, time, tWindow_ms, delta=1):
    
    w_length = len(np.arange(np.abs(time - 0).argmin(), np.abs(time - tWindow_ms).argmin()))
    
    w_overlap = math.floor(w_length * delta)
    
    nTrials = X.shape[0]
    trial_length = X.shape[2]

    y_aug = []
    X_aug = []
    trial_indices = []
    time_indices = []
    for iTrial in range(nTrials):

        for iTime in range(0, trial_length, w_overlap):

            w_end = iTime + w_length
            
            if w_end >= trial_length:
                break

            X_aug.append(X[iTrial, :, iTime:w_end, :])
            y_aug.append(y[iTrial, :])
            trial_indices.append(trial_idx[iTrial])
            time_indices.append(iTime)
    
    X_aug = np.asarray(X_aug)
    y_aug = np.asarray(y_aug)

    return X_aug, y_aug, trial_indices, time_indices


def custom_cv_balance_split(labels, nFolds=5, test_size=0.2, random_state=0):
    
    np.random.seed(random_state)
    
    # get frequency of each label
    uniq_labels, counts = np.unique(labels, return_counts=True)
    
    # determine number of trials for validation
    nTest_trials = math.ceil(min(counts)*test_size)

    counts -= nTest_trials
    
    min_trials = min(counts) - 1
    
    # get trial indices for each label
    label_idxs = [np.argwhere(labels == i).reshape(-1) for i in uniq_labels]

    # loop over folds 
    for iFold in range(nFolds):

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
    


def kf_training(X, y, nEpochs, batch_size, cv, nChans, kernLength, time, call_backs,
               tWindow_ms, tWindow_overlap):

    # create test set of whole epochs (prevents bleeding of training data into test data)
    test_size = math.ceil(X.shape[0]*0.1)
    #splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    #train_val_idx, test_idx = next(splitter.split(X, y))
    
    splitter = cv(y, nFolds=1, test_size=0.1)
    train_val_idx, test_idx = next(splitter)
    
    nClasses = len(np.unique(y))
    
    # convert labels to one hot encoding (easier than changing the last layer of EEGNet to sigmoid)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    y = enc.fit_transform(y[:,np.newaxis])
    
    X_test = X[test_idx,:]
    y_test = y[test_idx,:]
    
    X_train_val = X[train_val_idx,:]
    y_train_val = y[train_val_idx,:]
    
    # train on data only ~1000ms prior to shot initiation to reduce noise
    train_val_data_startTime = 0 #ms
    train_val_data_startIdx = np.where(time == train_val_data_startTime)[0][0]
    
    X_train_val = X_train_val[:,:, train_val_data_startIdx:, : ]
    train_val_time = time[train_val_data_startIdx:]
    
    fold = 0
    folds_valAcc = []
    folds_valLoss = []
    folds_valAuc = []
    folds_valF1 = []
    
    nFolds = 5
    train_val_splitter = cv(y_train_val.argmax(axis = -1), nFolds=nFolds, test_size=0.2)
    
    
    #for train_idx, val_idx in cv.split(X_train_val, y_train_val.argmax(axis= -1)):
    for iFold in range(nFolds):
        
        train_idx, val_idx = next(train_val_splitter)
        
        X_train = X_train_val[train_idx, :]
        y_train = y_train_val[train_idx, :]
        
        X_val = X_train_val[val_idx, :]
        y_val = y_train_val[val_idx, :]
        
        # account for imbalance in makes and misses for each subject
        unique_labels, counts = np.unique(y_train.argmax(axis=-1), return_counts=True)
        class_weights = dict(zip(unique_labels, (1./counts) * (len(y_train.argmax(axis=-1))/2.0) ))
    
    
        # apply data augmentation
        X_train, y_train, train_augIdx, train_time = slidingCrops_DA(X_train, y_train, train_idx, 
                                                                     train_val_time, tWindow_ms, tWindow_overlap)
        
        X_val, y_val, val_augIdx, val_time = slidingCrops_DA(X_val, y_val, val_idx, 
                                                                     train_val_time, tWindow_ms, tWindow_overlap)
        
        # clear model to avoid consuming memory since creating new model with each new fold
        K.clear_session()
           
        # initialize HyperModel version of EEGNet
        model = Hyper_EEGNet(nClasses = nClasses, 
                             nSamples = X_train.shape[2], 
                             nChans = nChans, 
                             kernLength = kernLength)
        
        tDelta_ms = int(math.floor(tWindow_ms * tDelta))
        
        # pass model object into tuner. use validation f1 score to account for imbalance
        tuner = kt.Hyperband(model, 
                             objective = kt.Objective('val_f1_score', direction='max'), 
                             max_epochs = 30, 
                             factor = 3,
                             directory=modelDir,
                             project_name=f'sj{sjNum}_EEGNet/HyperbandOpt_{tWindow_ms}_{tDelta_ms}',
                             overwrite=True)
        
        # search paramater space. all arguments are passed to the models.fit() method
        tuner.search(X_train, y_train, batch_size = batch_size, epochs = 100,
                     validation_data = (X_val, y_val),
                     class_weight = class_weights,
                     callbacks=[call_backs],
                     verbose = 1)
        
        # print summary of tuning results
        tuner.results_summary()
        
        # get best hyperparameters and retrain model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        model = tuner.hypermodel.build(best_hps)
        
        
        # configure the EEGNet model with kernel length of 1/2 sampling rate
        #model = EEGNet(nb_classes = len(np.unique(y.argmax(axis=-1))), Chans = nChans, Samples = X_train.shape[2], 
                           #dropoutRate = 0.5, kernLength = kernLength, F1 = F1, D = D, F2 = F1*D, 
                           #dropoutType = 'Dropout')

        # compile the model and set the optimizers
        #model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate = lr), 
                      #metrics = ['accuracy'])
        
        # Fit initial pass of model
        init_history = model.fit(X_train, y_train, batch_size = batch_size, epochs = nEpochs,
                            validation_data=(X_val, y_val), 
                            class_weight=class_weights,
                            callbacks = [call_backs],
                            verbose = 1)
        
        # get optimal number of epochs for training
        val_acc_per_epoch = init_history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print(f'Best epoch: {best_epoch}')
        
        # Fit second pass of model using optimal number of epochs
        model = tuner.hypermodel.build(best_hps)
        
        history = model.fit(X_train, y_train, batch_size = batch_size, epochs = best_epoch,
                            validation_data=(X_val, y_val), 
                            class_weight=class_weights,
                            callbacks = [call_backs],
                            verbose = 1)
        
        
        train_acc = history.history['accuracy'][0]
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        val_auc = history.history['val_auc'][0]
        val_f1 = history.history['val_f1_score'][0]
        
        val_preds = model.predict(X_val).argmax(axis=1)
        val_true = y_val.argmax(axis=1)

        val_acc = balanced_accuracy_score(val_true, val_preds)
        
        fold += 1

        print(f'\nFold {fold}/{nFolds} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_acc:.2%} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}\n')


        # check if this model performed better than others
        if folds_valAuc and val_auc > max(folds_valAuc):
            best_model = model
            print(f'Updated Best Model. Validation loss decreased from {min(folds_valLoss):.3f} to {val_loss:.3f}.\n')
        elif not folds_valAuc:
            best_model = model


        folds_valLoss.append(val_loss)
        folds_valAcc.append(val_acc)
        folds_valAuc.append(val_auc)
        folds_valF1.append(val_f1)
            
            
    folds_valLoss = np.asarray(folds_valLoss)
    folds_valAcc = np.asarray(folds_valAcc)
    folds_valAuc = np.asarray(folds_valAuc)
    folds_valF1 = np.asarray(folds_valF1)

    model_saveFolder = os.path.join(modelDir, f'sj{sjNum}_EEGNet')
    model_saveName = os.path.join(model_saveFolder, f'HyperBand_EEGNet_{tWindow_ms}_{tDelta_ms}')
    
    if ~os.path.exists(model_saveName):
        os.mkdir(model_saveName)
    
    # save best model
    best_model.save(model_saveName)
    
    
    # apply best model to test data
    w_length = len(np.arange(np.abs(time - 0).argmin(), np.abs(time - tWindow_ms).argmin()))
    
    w_overlap = math.floor(w_length * tDelta)
    
    time_acc = []
    time_auc = []
    
    test_true = y_test.argmax(axis=1)
    
    for iTime in range(0, len(time), w_overlap):
        
        w_end = iTime + w_length
        
        if w_end > len(time):
            break
        
        test_preds = best_model.predict(X_test[:,:, iTime:w_end, :], verbose=0).argmax(axis=1)
        
        test_acc = balanced_accuracy_score(test_true, test_preds)
        test_auc = roc_auc_score(test_true, test_preds)
        
        time_acc.append(test_acc)
        time_auc.append(test_auc)
        
        
    time_acc = np.asarray(time_acc)

    cv_results = {}
    cv_results['Folds'] = range(kf.n_splits)
    cv_results['Val Loss'] = folds_valLoss
    cv_results['Val Acc'] = folds_valAcc * 100
    cv_results['Val Auc'] = folds_valAuc
    cv_results['Test Acc'] = time_acc * 100
    cv_results['Test AUC'] = time_auc
    
    return cv_results
    

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--sjNum", type=int, help='Subject Number')
    parser.add_argument("--nEpochs", type=int, default=10, help="Number of Epochs for training, default=10")
    parser.add_argument("--nFolds", type=int, default=3, help="Number of Folds for cv, default=3")
    #parser.add_argument("--F1", type=int, default=8, help='Number of Temporal Filters')
    #parser.add_argument("--D", type=int, default=2, help='Number of Spatial Filters')
    parser.add_argument("--tWindow", type=int, default=1000, help='Duration of data crops (ms)')
    parser.add_argument("--tDelta", type=float, default=0.5, help='Percent of overlap in crops')
    
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
    
    tWindow_ms = args.tWindow # define time window size for sliding estimator
    
    tDelta = args.tDelta # size of each time step

    kf = StratifiedKFold(n_splits=args.nFolds, shuffle=True, random_state=0)
    
    #F1, D = args.F1, args.D
    nEpochs = args.nEpochs
    batch_size = 64
    
    early_stopping = EarlyStopping(monitor='val_loss', 
                                  patience=100,
                                  mode='min',
                                  restore_best_weights=True)
    
    call_backs = [early_stopping]
    
    # Cross validation training
    training_results = kf_training(X, y, nEpochs, batch_size, custom_cv_balance_split, nChans, 
                                   kernLength, time, call_backs, 
                                   tWindow_ms = tWindow_ms, tWindow_overlap=tDelta)
    
    tWindow_overlap = int(math.floor(tWindow_ms * tDelta))
    
    saveFile_name = os.path.join(saveDir, f'sj{sjNum:02d}_EEGNet_DA_optim_{tWindow_ms}_{tWindow_overlap}_results.pickle')
    
    with open(saveFile_name, 'wb') as handle:
        pickle.dump(training_results, handle)