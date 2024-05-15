import numpy as np
import math
import os
import copy
import h5py
from scipy.io import loadmat
import time
from datetime import datetime
import ast
import pickle

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, f1_score, recall_score, make_scorer

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from mne.decoding import CSP, Scaler
from scipy.signal import hilbert

import argparse

import mne

import sys

from tqdm import tqdm


################### Helper Classes/Functions ###########################

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
                 cov_est = self.cov_est)
        
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
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sjNum", type=int, help='Subject Number')
    parser.add_argument("--nFolds", type=int, default=3, help="Number of Folds for cv, default=3")
    parser.add_argument("--tWindow", type=int, default=250, help='Duration of data crops (ms)')
    parser.add_argument("--tDelta", type=int, default=125, help='Duration of overlap in crops (ms)')
    parser.add_argument("--comps", type=int, default=4, help='Number of CSP components')
    
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

    
    ##################### Apply Classifiyer ######################

    
    data = np.asarray(data, dtype=float)
    
    
    freqs = [(4,7), (8,12), (12, 30)]
    
    # Filter data
    X_filt = np.empty((len(freqs), data.shape[0], data.shape[1], data.shape[2]))

    for i, freq_band in enumerate(freqs):

        # filter activity within specific frequency band
        X_filt[i,:] = mne.filter.filter_data(data, srate, 
                                             l_freq = freq_band[0], h_freq = freq_band[1],
                                             method = 'iir', iir_params = {'order': 3, 'ftype': 'butter'})

    X_filt = X_filt.swapaxes(0,1)
    
    X_hilb = hilbert(X_filt, axis = -1)
    
    X_pow = np.abs(X_hilb)**2
    
    nComponents = args.comps
    
    custom_csps = Custom_CSP(n_components=nComponents, freq_bands=freqs, 
                             norm_trace=False, log=True, cov_est='epoch')
    
    
    tWindow_ms = args.tWindow # define time window size for sliding estimator
    tWindow_size = len(np.arange(np.abs(time - 0).argmin(), 
                                 np.abs(time - tWindow_ms).argmin()
                                )
                      )

    tWinOverlap_ms = args.tDelta
    
    winOverlap = tWinOverlap_ms / tWindow_ms
    
    tOverlap_step = math.floor(tWindow_size * (1-winOverlap))
    
    # Split data in to Training/ Test sets
    y = np.asarray(shotLabels, dtype=np.int64) # make sure labels are integers
    #test_size = math.ceil(data.shape[0]*0.1)
    test_size = 0.1
    
    #splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    #train_val_idx, test_idx = next(splitter.split(X_pow, y))
    
    splitter = custom_cv_balance_split(y, nFolds=1, test_size=test_size) # outputs list of (train, test) indices
    train_val_idx, test_idx = next(splitter)

    X_train = X_pow[train_val_idx, :]
    y_train = y[train_val_idx]

    X_test = X_pow[test_idx, :]
    y_test = y[test_idx]

    class_priors =  1 - (np.bincount(y_train) / X_train.shape[0])
    #class_priors = [0.55, 0.45]
    
    LDA = LinearDiscriminantAnalysis(priors=class_priors)
    svm = SVC(kernel='linear', C=1, gamma='scale')

    clf = make_pipeline(custom_csps,
                        StandardScaler(),
                         LDA)
    
    # define hyperparameters for bayesian optimization
    svd_params = {
        'custom_csp__n_components': Integer(1, 4),
        'lineardiscriminantanalysis__solver': Categorical(['svd'])
    }

    shrink_params = {
        'custom_csp__n_components': Integer(1, 4), 
        'lineardiscriminantanalysis__solver': Categorical(['lsqr', 'eigen']),
        'lineardiscriminantanalysis__shrinkage': Real(0., 1.0, prior='uniform')
    }

    #cv = StratifiedKFold(n_splits=args.nFolds, shuffle=True, random_state=0)

    train_val_splitter = custom_cv_balance_split(y_train, nFolds=args.nFolds, test_size=0.2)
    
    cv = [next(train_val_splitter) for i in range(args.nFolds)]
    
    # define bayes optimization function
    opt = BayesSearchCV(
        clf,
        [(svd_params, 30), (shrink_params, 60)],
        cv = cv,
        scoring = 'accuracy',
        n_jobs = -1
    ) 
    
    specificity = make_scorer(recall_score, pos_label=0)

    scoring_metrics = {
        'accuracy': make_scorer(accuracy_score),
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'precision': make_scorer(precision_score),
        'recall' : make_scorer(recall_score),
        'specificity': specificity,
        'auc' : make_scorer(roc_auc_score),
        'f1' : make_scorer(f1_score)
    }

    
    time_trainAcc = []
    time_train_balAcc = []
    time_trainSpecificity=[]
    time_trainRecall = []
    
    time_valAcc = []
    time_val_balAcc = []
    time_valAuc = []
    time_valF1 = []
    time_valPrecision = []
    time_valRecall = []
    time_valSpecificity = []


    time_testAcc = []
    time_test_balAcc = []
    time_testAuc = []
    time_testF1 = []
    time_testPrecision = []
    time_testRecall = []
    time_testSpecificity = []
    
    mne.set_log_level('WARNING')
    
    print('Training model...')
    
    for iTime in tqdm(range(0, len(time)-tWindow_size, tOverlap_step)):

        tend_idx = iTime + tWindow_size

        # extract training data for this time window
        X = X_train[:, :, :, iTime:tend_idx]

        # First optimize over hyper parameters w/ Bayes
        #opt.fit(X, y_train)
        
        # extract model
        #best_optModel = opt.best_estimator_
        
        # Refit data using optimized model to get other scoring metrics
        cv_scores = cross_validate(clf, X, y_train,
                                   scoring=scoring_metrics,
                                   cv = cv, 
                                   return_train_score=True,
                                   return_estimator=True)
        
        # extract cross validation metrics
        cv_trainAcc = cv_scores['train_accuracy']
        cv_train_balAcc = cv_scores['train_balanced_accuracy']
        cv_trainSpecificity = cv_scores['train_specificity']
        cv_trainRecall = cv_scores['train_recall']
        
        cv_valAcc = cv_scores['test_accuracy']
        cv_val_balAcc = cv_scores['test_balanced_accuracy']
        cv_auc = cv_scores['test_auc']
        cv_f1 = cv_scores['test_f1']
        cv_precision = cv_scores['test_precision']
        cv_specificity = cv_scores['test_specificity']
        cv_recall = cv_scores['test_recall']

        time_trainAcc.append(cv_trainAcc.mean())
        time_train_balAcc.append(cv_train_balAcc.mean())
        time_trainSpecificity.append(cv_trainSpecificity.mean())
        time_trainRecall.append(cv_trainRecall.mean())
        
        time_valAcc.append(cv_valAcc.mean())
        time_val_balAcc.append(cv_val_balAcc.mean())
        time_valAuc.append(cv_auc.mean())
        time_valF1.append(cv_f1.mean())
        time_valPrecision.append(cv_precision.mean())
        time_valRecall.append(cv_recall.mean())
        time_valSpecificity.append(cv_specificity.mean())


        # extract best model
        bestMdl_idx = cv_valAcc.argmax()

        bestMdl = cv_scores['estimator'][bestMdl_idx]
        #bestMdl = opt.best_estimator_

        test_preds = bestMdl.predict(X_test[:, :, :, iTime:tend_idx])

        test_acc = accuracy_score(y_test, test_preds)
        test_balAcc = balanced_accuracy_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds)
        test_recall = recall_score(y_test, test_preds)
        test_specificity = specificity(bestMdl, X_test, y_test)


        time_testAcc.append(test_acc)
        time_test_balAcc.append(test_balAcc)
        time_testAuc.append(test_auc)
        time_testF1.append(test_f1)
        time_testPrecision.append(test_precision)
        time_testRecall.append(test_recall)
        time_testSpecificity.append(test_specificity)
    
    tWindow_overlap_ms = int(math.floor(tWindow_ms * winOverlap))
    
    training_results = {}
    training_results['time'] = np.linspace(time[0], 
                                        time[tend_idx], 
                                        len(time_trainAcc))
    
    training_results['Train'] = dict(zip(['accuracy',
                                          'balanced_accuracy',
                                          'recall',
                                          'specificity'],
                                         [time_trainAcc,
                                          time_train_balAcc,
                                          time_trainRecall,
                                          time_trainSpecificity]
                                        )
                                    )
    
    metric_keys = ['accuracy', 'balanced_accuracy', 'auc', 'f1', 'recall', 'specificity']
    
    training_results['Validation'] = dict(zip(metric_keys,
                                          [time_valAcc,
                                           time_val_balAcc,
                                           time_valAuc,
                                           time_valF1,
                                           time_valRecall,
                                           time_valSpecificity]
                                             )
                                         )
    
    
    training_results['Test'] = dict(zip(metric_keys,
                                          [time_testAcc,
                                           time_test_balAcc,
                                           time_testAuc,
                                           time_testF1,
                                           time_testRecall,
                                           time_testSpecificity]
                                       )
                                   )
    
    
    print(f'Average Validation Accuracy: {np.mean(time_valAcc) * 100:.2f}%')
    
    saveFile_name = os.path.join(saveDir, f'sj{sjNum:02d}_CSP_{tWindow_ms}_{tWinOverlap_ms}_results.pickle')
    
    with open(saveFile_name, 'wb') as handle:
        pickle.dump(training_results, handle)