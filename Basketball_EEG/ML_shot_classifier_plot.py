import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
from scipy.stats import sem


if __name__ == '__main__':
    
    parentDir = '/home/bullock/Gimmick_Ball/Jordan'
    dataDir = os.path.join(parentDir, 'Classifier_Results')
    figDir = os.path.join(parentDir, 'Figures')

    subjects = np.hstack([np.arange(11,15), np.arange(16,40), np.arange(51,57)])
    
    #subjects = np.setdiff1d(subjects,[30,39,52])

    file_suffix = 'CSP_250_125_results.pickle'
    
    allSj_testAcc = []
    allSj_testRecall = []
    allSj_testSpecificity = []
    
    allSj_valAcc = []
    allSj_valRecall = []
    allSj_valSpecificity = []
    
    allSj_trainAcc = []
    allSj_trainRecall = []
    allSj_trainSpecificity = []
    
    # Compile classifier results
    for iSub, sjNum in enumerate(subjects):
        
        sjFile_name = os.path.join(dataDir, f'sj{sjNum:02d}_{file_suffix}')
        
        with open(sjFile_name, 'rb') as handle:
            sj_classifier_results = pickle.load(handle)
            
        train_results = sj_classifier_results['Train']
        val_results = sj_classifier_results['Validation']
        test_results = sj_classifier_results['Test']
        
        
        allSj_valAcc.append(val_results['accuracy'])
        allSj_valRecall.append(val_results['recall'])
        allSj_valSpecificity.append(val_results['specificity'])
        
        allSj_testAcc.append(test_results['accuracy'])
        allSj_testRecall.append(test_results['recall'])
        allSj_testSpecificity.append(test_results['specificity'])
                                    
        allSj_trainAcc.append(train_results['accuracy'])
        allSj_trainRecall.append(train_results['recall'])
        allSj_trainSpecificity.append(train_results['specificity'])
        
                              
                              
    # convert lists into matrices
    allSj_valAcc = np.vstack(allSj_valAcc) * 100.
    allSj_valRecall = np.vstack(allSj_valRecall) * 100.
    allSj_valSpecificity = np.vstack(allSj_valSpecificity) * 100.
    
    allSj_testAcc = np.vstack(allSj_testAcc) * 100.
    allSj_testRecall = np.vstack(allSj_testRecall) * 100.
    allSj_testSpecificity = np.vstack(allSj_testSpecificity) * 100.
    
    allSj_trainAcc = np.vstack(allSj_trainAcc) * 100.
    allSj_trainRecall = np.vstack(allSj_trainRecall) * 100.
    allSj_trainSpecificity = np.vstack(allSj_trainSpecificity)* 100.
    
    ###### Plot Results ######
    plt_time = sj_classifier_results['time']
    
    avg_valAcc = np.mean(allSj_valAcc, axis=0)
    avg_valRecall = np.mean(allSj_valRecall, axis=0)
    avg_valSpecificity = np.mean(allSj_valSpecificity, axis=0)
    
    sem_valAcc = sem(allSj_valAcc, axis=0)
    sem_valRecall = sem(allSj_valRecall, axis=0)
    sem_valSpecificity = sem(allSj_valSpecificity, axis=0)
    
    avg_testAcc = np.mean(allSj_testAcc, axis=0)
    avg_testRecall = np.mean(allSj_testRecall, axis=0)
    avg_testSpecificity = np.mean(allSj_testSpecificity, axis=0)
    
    sem_testAcc = sem(allSj_testAcc, axis=0)
    sem_testRecall = sem(allSj_testRecall, axis=0)
    sem_testSpecificity = sem(allSj_testSpecificity, axis=0)
    
    avg_trainAcc = np.mean(allSj_trainAcc, axis=0)
    avg_trainRecall = np.mean(allSj_trainRecall, axis=0)
    avg_trainSpecificity = np.mean(allSj_trainSpecificity, axis=0)
    
    sem_trainAcc = sem(allSj_trainAcc, axis=0)
    sem_trainRecall = sem(allSj_trainRecall, axis=0)
    sem_trainSpecificity = sem(allSj_trainSpecificity, axis=0)
    
    fig, ax = plt.subplots(nrows=3, figsize = (11,8), dpi=100)
    
    
    metric_labels = ['Test', 'Validation', 'Train']
    
    for i in range(3):
        
        if not i:
            metric1 = avg_testAcc 
            metric2 = avg_valAcc
            metric3 = avg_trainAcc
            
            metric1_sem = sem_testAcc
            metric2_sem = sem_valAcc
            metric3_sem = sem_trainAcc
            
            ylabel = 'Accuracy'
        elif i == 1:
            
            metric1 = avg_testRecall
            metric2 = avg_valRecall
            metric3 = avg_trainRecall
            
            metric1_sem = sem_testRecall
            metric2_sem = sem_valRecall
            metric3_sem = sem_trainRecall
            
            ylabel = 'Recall'
        else:
            
            metric1 = avg_testSpecificity
            metric2 = avg_valSpecificity 
            metric3 = avg_trainSpecificity
            
            metric1_sem = sem_testSpecificity
            metric2_sem = sem_testSpecificity
            metric3_sem = sem_testSpecificity
            
            ylabel = 'Specificity'
            
        # mean traces
        ax[i].plot(plt_time, metric1, label=metric_labels[0], lw=1.5)
        ax[i].plot(plt_time, metric2, label=metric_labels[1], lw=1.5)
        
        ax[i].plot(plt_time, metric3, label=metric_labels[2], lw=1.5)
        
        # sem of mean traces
        ax[i].fill_between(plt_time, metric1-metric1_sem,
                        metric1+metric1_sem, alpha=0.15)
        
        ax[i].fill_between(plt_time, metric2-metric2_sem,
                        metric2+metric2_sem, alpha=0.15)
        
        ax[i].fill_between(plt_time, metric3-metric3_sem,
                        metric3+metric3_sem, alpha=0.15)
        
        
       
        ax[i].set_ylim([40.,90.])
            
        ax[i].set_xlim([plt_time[0], plt_time[-1] + 250])
        ax[i].set_ylabel(ylabel) 
        
        if not i:
            legend = ax[i].legend(ncols=3)
        else:
            legend = ax[i].legend(ncols=2)
            
        handles = legend.legendHandles
        ax[i].legend(handles=[handles[i] for i in range(len(metric_labels))], 
          labels=metric_labels, ncol=len(metric_labels), numpoints=1, frameon=False)
        
        ax[i].axvline(.0, color='k', linestyle='-')
        ax[i].axhline(50., color='k', linestyle='--', label='chance')
        
        if i == 2:
            ax[i].set_xlabel('Time (ms)')
        else:
            ax[i].set_xticks([])
            
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        

    plt.savefig(os.path.join(figDir, f'CSP_250_classifier_performance.jpg'), 
                bbox_inches='tight')