function EEG_preprocessing2 (subjects, exerCon, showreject_trials,processForIEM, check_topo)

%==========================================================================
%{ 
wtf_eegProcessing2
Purpose:Convert EEG and Behavioral data to the same format as the data in 
Foster et al. (2016), thus allowing us to use their scripts for data 
processing.  Also, reject noisy channels (manual).
Author: Tom Bullock, UCSB Attention Lab
Date Created: 4.4.16
Date Last Modified: 3.3.18

Processed using EEGLAB v13.0.1
Must have already run wtf_eegPreprocessing.m
%}
%==========================================================================



cd('/home/garrett/eeglab14_1_2b')
eeglab
close all


% what processing type? 1=IEM (remove noisy chans), 2=HILBERT (interpolate noisy chans)


sourceFolder = '/home/garrett/WTF_Bike/EEG_ft_ep_all'; % filtered, epoched files
destFolder_beh = '/home/garrett/WTF_Bike/Data'; % behavioral data

if processForIEM==1;destFolder_EEG = '/home/garrett/WTF_Bike/EEG'; % files with noisy channels removed for IEM processing
elseif processForIEM==2;destFolder_EEG = '/home/garrett/WTF_Bike/HILBERT'; % files with noisy channels interpolated for POWER plots etc...
end

% which subjects?
%subjects = [1];

%matlabpool open 72
for iSub=1:length(subjects)
    
    % set subject
    sjNum=subjects(iSub);
    disp(['Processing subject : ' num2str(sjNum)])
    
    %% load original filtered EEG file (broken trials + mismatched trials already taken care of)
    EEG = pop_loadset([sourceFolder '/' sprintf('sj%02d_exerCond%02d_changeDect_wmTF_bl01_ft_ep.set',sjNum, exerCon)]);
    
    
    %% extract behavior from my epoched EEG files
    
    %change detection needs hits (correct detection of change) and false
    %alarms (incorrect detection of change)
    
    beh = [];
    hits = 0;
    false_alarms = 0;
    hits_total = 0;
    false_alarms_total=0;
    correct_rejections = 0;
    miss = 0;
    for iBeh = 1:length(EEG.newTrialInfo)
        beh.trial.pos(iBeh) = EEG.newTrialInfo(iBeh).stimLocAngle;
        beh.trial.posBin(iBeh) = EEG.newTrialInfo(iBeh).stimLoc;
        %beh.trial.err(iBeh) = abs(EEG.newTrialInfo(iBeh).stimLocAngle - EEG.newTrialInfo(iBeh).thisMouseAngleDegs);
        beh.trial.respPos(iBeh) = EEG.newTrialInfo(iBeh).RespStimLocAngle;
        beh.trial.block(iBeh) = EEG.newTrialInfo(iBeh).thisBlock;
        
        % counting hits and false alarms for d'
        if beh.trial.pos(iBeh) == beh.trial.respPos(iBeh) % no change
            if EEG.newTrialInfo(iBeh).ChangeResp == 1 %correct response
                beh.trial.err(iBeh) = 0;
                beh.trial.hit(iBeh) = 0;
                beh.trial.correct_reject(iBeh) = 1;
                beh.trial.false_alarm(iBeh) = 0;
                beh.trial.miss(iBeh) = 0;
                
                correct_rejections = correct_rejections + 1;
                false_alarms_total = false_alarms_total + 1;
            else %incorrect response
                beh.trial.err(iBeh) = 1;
                beh.trial.false_alarm(iBeh) = 1;
                beh.trial.hit(iBeh) = 0;
                beh.trial.correct_reject(iBeh) = 0;
                beh.trial.miss(iBeh) = 0;
                
                false_alarms = false_alarms + 1;
                false_alarms_total = false_alarms_total + 1;
            end
        elseif beh.trial.pos(iBeh) ~= beh.trial.respPos(iBeh) % change
            if EEG.newTrialInfo(iBeh).ChangeResp == 1 %incorrect response
                beh.trial.err(iBeh) = 1;
                beh.trial.miss(iBeh) = 1;
                beh.trial.hit(iBeh) = 0;
                beh.trial.false_alarm(iBeh) = 0;
                beh.trial.correct_reject(iBeh) = 0;
                
                
                miss = miss + 1;
                hits_total = hits_total + 1;
            else %correct response
                beh.trial.err(iBeh) = 0;
                beh.trial.miss(iBeh) = 0;
                beh.trial.hit(iBeh) = 1;
                beh.trial.false_alarm(iBeh) = 0;
                beh.trial.correct_reject(iBeh) = 0;
                
                hits = hits + 1;
                hits_total = hits_total + 1;
            end
        end
    end
    
    beh.hits = hits;
    beh.false_alarms=false_alarms;
    beh.hits_total = hits_total;
    beh.false_alarms_total= false_alarms_total;
    beh.correct_rejections = correct_rejections;
    beh.misses = miss;
    
    %% extract EEG data from my epoched EEG files
    eeg = []; % create eeg struct
    
    % get rid of problem channels on a subject by subject basis (determined
    % by visual inspection).  To ensure no bias, if an electrode is faulty
    % in one condition it gets removed for all conditions for that subject.
    if processForIEM ==1
        if sjNum == 1
            EEG = pop_select(EEG, 'nochannel', {'O2','C5','P5','AF8','PO4','FC5'});
            
        elseif sjNum == 4
            EEG = pop_select(EEG, 'nochannel', {'O2','PO4'});
            
        elseif sjNum == 8
            EEG = pop_select(EEG, 'nochannel', {'CP3','F4','C2','FT8','F8','FC4','CP1'});
            
        elseif sjNum == 12
            EEG = pop_select(EEG, 'nochannel', {'F7'});
            
        elseif sjNum == 23
            EEG= pop_select(EEG, 'nochannel', {'FC6','AF7','TP8','Fp1','FT8','F2','F6','Fp2','FC5'});
        
        elseif sjNum == 27
            EEG = pop_select(EEG,'nochannel',{'T7'});
            
        elseif sjNum == 32
            EEG= pop_select(EEG,'nochannel', {'F8','CP2'});
            
        elseif sjNum == 33
            EEG = pop_select(EEG, 'nochannel',{'Pz','FC6','C5','F5','TP8'});
            
        elseif sjNum == 34
            EEG = pop_select(EEG, 'nochannel', {'FT10'});
            
        end
        
        %continous WM Task rejections
%         if sjNum == 2
%             EEG = pop_select(EEG,'nochannel',{'C2'});
%         elseif sjNum == 3
%             EEG =pop_select(EEG, 'nochannel',{'T7'});
%         elseif sjNum == 4
%             EEG = pop_select(EEG,'nochannel',{'AF4','AFz'});
%         end

    elseif processForIEM==2
        bad_electrodes = {};
        %define electrodes to interpolate
        if sjNum == 1
            bad_electrodes = {'O2','C5','P5','AF8','PO4','FC5','P8','O1','Oz'};
        elseif sjNum == 2
            bad_electrodes = {'P8','P5','CP4','O1','P6','CP2','P7','PO7','PO8','Pz'};
        elseif sjNum == 3
            bad_electrodes = {'AF8','P5','O2'};
            
        elseif sjNum == 4 
            bad_electrodes =  {'O2','PO4','Fp2','C5','Pz','P6','P4','P3','P7','P2','P5'};
            
        elseif sjNum == 5
            bad_electrodes = {'T7','PO4','O2','Oz'};
            
        elseif sjNum == 6 
            bad_electrodes = {'PO3','Pz','PO7','POz','CP3'};
            
        elseif sjNum == 7 
            bad_electrodes = {'PO3','POz','CP4','P8','T7','Pz'};
            
        elseif sjNum == 8 %bad
            bad_electrodes = {'CP3','F4','C2','FT8','F8','FC4','CP1','AF8'};%{'CP3','F4','C2','FT8','F8','FC4','CP1','AF8','PO4','FT9','TP9','TP10','Fp2','F7','FT7'};
            
        elseif sjNum == 10 
            bad_electrodes = {'FC6','AF8'};
            
        elseif sjNum == 11
            bad_electrodes = {'PO3','POz','FC6','TP8','P4'}; 
            
        elseif sjNum == 12 
            bad_electrodes = {};
        
        elseif sjNum == 13
            bad_electrodes = {'PO4'};
            
        elseif sjNum == 14 %probably dont include in topoplots
            bad_electrodes = {'P5','PO7','P1','P8','PO8','P3','O1','P7','POz','PO4',...
                'P4','CP3','PO3','CP4','CP5','Oz','O2','P6'};%{'P5','PO7','P1','P8','P3','PO8','O1'}; %huge numbers
            
        elseif sjNum == 15 
            bad_electrodes = {'CPz','P4','PO7','PO8','O1'};
            
        elseif sjNum == 16 
            bad_electrodes = {'PO3','PO7','POz'};
            
        elseif sjNum == 17
            bad_electrodes = {};
            
        elseif sjNum == 18 
            bad_electrodes = {'PO4','POz','O2','P5','PO3'};
            
        elseif sjNum == 19
            bad_electrodes = {'FT10','PO4','P6','Pz'};
        
        elseif sjNum == 20 
            bad_electrodes = {'C4','P3','Fp2','CP3'};
            
        elseif sjNum == 21 
            bad_electrodes = {'P5','O1','PO7','P1','P7','PO4','POz'};
            
        elseif sjNum == 22 
            bad_electrodes = {'AF8','Fp2','F8'};
           
        elseif sjNum == 23 
            bad_electrodes = {'FC6','AF7','TP8','Fp1','FT8','F2','F6','Fp2','FC5','PO4','O1','O2','P5','Oz','POz','P8','PO3','PO7','PO8'};
       
        elseif sjNum == 24
            bad_electrodes = {'AF8'};
        
        elseif sjNum == 25
            bad_electrodes = {};
            
        elseif sjNum == 26
            bad_electrodes = {'AF8'};
            
        elseif sjNum == 27 
            bad_electrodes = {'T7','Pz','AF8','PO4'};
            
        elseif sjNum == 28
            bad_electrodes = {};
            
        elseif sjNum == 29
            bad_electrodes = {'Pz','P4','P6'};
            
        elseif sjNum == 30
            bad_electrodes={'FT10'};
            
        elseif sjNum == 31
            bad_electrodes = {'POz','PO4','F3'};
            
        elseif sjNum == 32
            bad_electrodes = {'F8','CP2','CP3'};
            
        elseif sjNum == 33  
           bad_electrodes = {'Pz','FC6','C5','F5','TP8','PO7','PO3'}; 
           
        elseif sjNum == 34 
            bad_electrodes = {'FT10','AF8','P5','O2'};
            
        elseif sjNum == 35 %rerun
            bad_electrodes = {'PO3','Pz','CP3','P6','PO8','P8','PO4','CP2','P2','TP7','CP1'};%'PO4','P8','Pz','PO3','P2'
            
        end
        %get numerical location of electrodes
        badElectrodeIndex = [];
        if ~isempty(bad_electrodes)
            cnt=0;
            for iChan=1:length(EEG.chanlocs)
                if ismember(EEG.chanlocs(iChan).labels,bad_electrodes)
                    cnt=cnt+1;
                    badElectrodeIndex(cnt) = iChan;
                end
            end
            
            if length(bad_electrodes) ~= length(badElectrodeIndex)
                disp('NOT ALL CHANNELS INTERPOLATED!!!')
                return
            end
            
            EEG = pop_interp(EEG,badElectrodeIndex,'spherical');
        end
        
    end
    
    % remove reference channels (1&2) and EOG (3:8)
    EEG = pop_select(EEG,'nochannel',{'HEOG','EKG','ACC_X_HEAD','ACC_Y_HEAD','ACC_Z_HEAD','ACC_X_LEG','ACC_Y_LEG','ACC_Z_LEG'});

    % optional: restrict to 20 electrodes (similar to Foster et al. 2016)
    %%EEG = pop_select(EEG,'channel',{'F3','Fz','F4','C3','Cz','C4','P3','Pz','P4','O1','O2','POz','PO3','PO4','PO7','PO8','P7','P8','CP5','CP6'});
     
    % do threshold based artifact rejection on whole epoch, but don't actualy
    % remove trials (just log the indices of removed trials)
    if showreject_trials == 0
        [EEG]  = pop_eegthresh(EEG, 1, [1:EEG.nbchan], -150 ,150, -.5,2, 1, 0); % only [0.5-2secs]
    elseif showreject_trials == 1
        [EEG,rejectIndex]  = pop_eegthresh(EEG, 1, [1:EEG.nbchan], -150 ,150, -.5,2, 0, 0);
        plotRejThr=trial2eegplot(EEG.reject.rejthresh,EEG.reject.rejthreshE,EEG.pnts,EEG.reject.rejthreshcol);
        rejE=plotRejThr;
        %Draw the data.
        %eegplot(EEG.data,...
            %'eloc_file',EEG.chanlocs, ...
            %'srate',EEG.srate,...
            %'events',EEG.event,...
            %'winrej',rejE);
            pop_eegplot(EEG,1,1,0);
        return
    end
        eeg.arf.artIndCleaned = EEG.reject.rejthresh;
    
%     %remove rejected trials from eeg data   
%     for nTrial = 1:length(eeg.arf.artIndCleaned)
%         if eeg.arf.artIndCleaned(nTrial) == 1
%             EEG.newTrialInfo(nTrial) = [];
%         end
%     end
   
    eeg.newTrialInfo = EEG.newTrialInfo;
    
%     if length(eeg.newTrialInfo) == (length(EEG.epoch) - sum(eeg.arf.artIndCleaned))
%         disp('BEH and EEG data match!!')
%     else
%         disp('BEH and EEG data mismatch!!')
%         return
%     end
            
    % downsample WTF from 1024 Hz to 256 Hz (BRAIN PRODs 1000Hz > 250Hz)
    EEG = pop_resample(EEG,250);
    
    % extract data and convert to Foster et al. matrix format 
    % (trials x chans xsamples).  
    eeg.data = permute(EEG.data,[3,1,2]);
    
    % extract chan labels (excluding REF AND EOG)
    for iChans = 1:length(EEG.chanlocs)%-8
        eeg.chanLabels{iChans} = EEG.chanlocs(iChans).labels;
    end
    
    % add some extra info for ease of data processing
    eeg.preTime = EEG.xmin*1000;
    eeg.postTime = EEG.xmax*1000;
    eeg.sampRate = EEG.srate;
    eeg.chanInfo = EEG.chanlocs;
    eeg.times = EEG.times;
    eeg.pnts = EEG.pnts;
    eeg.epoch = EEG.epoch;
    
    %add artifact rejection index to behavioral data
    beh.artifacts =  eeg.arf.artIndCleaned;
    
    % save behavioral and eeg data matrices
    cd ('/home/garrett/WTF_Bike/Analysis_Scripts')
   
    if processForIEM == 2 && check_topo
        checkAlpha_Topo(sjNum,exerCon,EEG)
        return
    end
    
    %save for fieldtrip
    if processForIEM == 2
        EEG = pop_saveset(EEG,'filename',sprintf('sj%02d_exerCon%02d_hilbert.set',sjNum, exerCon),'filepath',destFolder_EEG);
    end
        
    
    parsave([destFolder_EEG '/' sprintf('sj%02d_exerCon%02d_changeDect_EEG.mat',sjNum, exerCon)],eeg)
    parsave([destFolder_beh '/' sprintf('sj%02d_exerCon%02d_changeDect_MixModel_wBias.mat',sjNum, exerCon)],beh)
    
end
%matlabpool close

clear all
close all
end