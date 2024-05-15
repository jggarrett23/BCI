function EEG_preprocessing1 (subjectNumbers, exerCon, filterData,synchEpochData,artifactCorrection)

%==========================================================================
%{ 
Pre-process WTF EEG data
Purpose: From raw EEG > Epoched data
Author: Tom Bullock, UCSB Attention Lab
Date Created: 4.4.16
Date Last Modified: 3.3.18

Processed using EEGLAB v13.0.1
Must have already merged behavioral data (EXPLAIN)
%}
%==========================================================================
processForIEM = 2;


cd('/home/garrett/eeglab14_1_2b')
eeglab 
close all

% seed rng
for rngShuffle =1:100
    rng('shuffle')
end

% which subject number(s) to process?
%subjectNumbers = [1, 2];
    
% decine which processing steps to run
% filterData=1;
% synchEpochData=1;
% artifactCorrection = 0; % artifact correction is a stage within this loop (keep switched on)
artifactRejectionThreshold=0;    disableThresholdRejection=0; % keep this off

% set directories for EEG (raw, filtered), Eye and Behavioral data
cdTmp = cd;

EEG_rawFolder = '/home/garrett/WTF_Bike/EEG_raw';
EEG_ftFolder = '/home/garrett/WTF_Bike/EEG_ft'; %set EEG_ft directory
%eyeSynchEEGfolder = '/home/bullock/Foster_WM/Eye_Synch_Folder'; %set eye-synching folder
%eyeFolder = '/home/bullock/Foster_WM/EYE_raw'; % set raw eye data folder
unbrokenTrialsFolder = '/home/garrett/WTF_Bike/Beh_Unbroken_Trials'; % set dir to store unbroken trials (for beh analysis)
behMergedFolder = '/home/garrett/WTF_Bike/Beh_merged';

%{
decide which prestimulus preprocessing to do: 
0=average baseline - use this for main spectral analyses, 
1= pre-stim baseline - use this for ERPs, EOG etc.
%}
whichPreprocessing=0;

if whichPreprocessing==0 % average baseline 
    EEG_ft_epFolder = '/home/garrett/WTF_Bike/EEG_ft_ep_all';              %set EEG_ft_ep directory
    thisBaseline = [-500 2000];                                             %set baseline correction (whole epoch)
    doHilbert=0;
    
    %%%%?????   
% elseif whichPreprocessing==1 % pre-stim baseline 
%     EEG_ft_epFolder = '/home/bullock/Foster_WM/PRESTIM_BL_EEG_ft_ep_all';   %set EEG_ft_ep directory
%     thisBaseline = [-500 0];                                                % set baseline correction (pre-stim)
%     doHilbert=0;                                                            % turn hilbert off  
end

% create some blank variables (these are for synch consistency checks)
misMatchLog = [];

% open cluster pool (max 144)
%matlabpool open 72

%loop through subjects


brkTrialVector=[]; EEG = [];                                            % create empty vars

%% set subject numbers and define exceptions (e.g. split EEG files due to bathroom breaks)
sjNum = subjectNumbers;
disp(['PROCESSING SUBJECT: ' num2str(sjNum)])
if sjNum==999
    mergedFile=1;
else
    mergedFile=0;
end

%% import/filter data routine
if filterData==1
    
    %cd into the raw data (edf) directorty
    cd(EEG_rawFolder)
    
    % merged or complete files (see "A_Merge_Broken_EEG_Files.m" script)
    if mergedFile==0
        %battery died during rest session for sj 15, had to start recording
        %again. Some data lost
        if sjNum == 15 && exerCon == 1
            d = dir(sprintf('sj%02d_exerCond01_changeDect_wmTF_bl01.vhdr',sjNum));
            EEG1= pop_fileio(d(1).name); %517 epochs
            d2 = dir(sprintf('sj%02d_exerCond01_changeDect_wmTF_bl01_11.vhdr',sjNum));
            EEG2=pop_fileio(d2(1).name); %199 epochs
            EEG=pop_mergeset(EEG1,EEG2);
        elseif sjNum == 23 && exerCon == 1
            d = dir(sprintf('sj%02d_exerCond01_changeDect_wmTF_bl01.vhdr',sjNum));
            EEG1= pop_fileio(d(1).name); %308 epochs
            d2 = dir(sprintf('sj%02d_exerCond01_changeDect_wmTF_bl01_2.vhdr',sjNum));
            EEG2=pop_fileio(d2(1).name); %491 epochs
            EEG=pop_mergeset(EEG1,EEG2);
        else
            d=dir(sprintf('sj%02d_exerCond%02d_changeDect_wmTF_bl01.vhdr',sjNum, exerCon));
            EEG = pop_fileio(d(1).name);
        end
    elseif mergedFile==1
        % gets filename for this subject & loads raw .set file
        d = dir(sprintf('sj%d.set',sjNum));
        EEG = pop_loadset(d(1).name);
    end
    
    % add channel locations
    EEG=pop_chanedit(EEG, 'lookup','/home/garrett/eeglab14_1_2b/plugins/dipfit2.3/standard_BESA/standard-10-5-cap385.elp'); % cluster
    %EEG=pop_chanedit(EEG,'lookup','/Users/tombullock1/Documents/MATLAB/ML_TOOLBOXES/eeglab13_0_1b/plugins/dipfit2.2/standard_BESA/standard-10-5-cap385.elp'); % local
    
    % reference to mastoids
    
    if sjNum == 10 %mastoids came off during exercise session, need to avg reference
        EEG = pop_select(EEG,'nochannel',{'HEOG','EKG','ACC_X_HEAD','ACC_Y_HEAD','ACC_Z_HEAD','ACC_X_LEG','ACC_Y_LEG','ACC_Z_LEG'});
        if processForIEM == 1
            EEG = pop_select(EEG, 'nochannel',{'TP9','TP10'});
        elseif processForIEM == 2
            EEG = pop_interp(EEG, [10 21],'spherical');
        end
        
        EEG = pop_reref(EEG, []);
    else
        EEG = pop_reref( EEG, [10 21],'keepref','on');
    end
    
    % filter
    EEG = pop_eegfiltnew(EEG,4,30); % DO THIS FOR EOG
    
    % *subject exceptions*
    if sjNum==999 % eyetracker froze at start of bl08, had to restart display stuff
        EEG.event(3002:3010) = [];
        EEG.urevent(3002:3010) = [];
    end
    
    if sjNum==999 % adds a 1 as the first event code (this was missing in the EEG data)
        EEG = pop_editeventvals(EEG,'insert',{1 [] [] []},'changefield',{1 'type' 1},'changefield',{1 'latency' 1});
    end
    
    % put electrodes in wrong positions
    if sjNum == 22
        EEG_temp = EEG;
        EEG.chanlocs(1:32) = EEG_temp.chanlocs(33:64);
        EEG.data(1:32,:,:) = EEG_temp.data(33:64,:,:);
        
        EEG.data(33:64,:,:) = EEG_temp.data(1:32,:,:);
        EEG.chanlocs(33:64) = EEG_temp.chanlocs(1:32);
        
    end
    
    % save filtered data
    EEG = pop_saveset(EEG,'filename',sprintf('%s_ft.set',d(1).name(1:end-5)),'filepath',EEG_ftFolder);
    
    %cd back to main directory
    cd('/home/garrett/WTF_Bike')
    
end

%% epoch EEG data and synch with MAT file (trial info)
if synchEpochData==1
    
    % gets filtered data filename for this subject
    d = dir([EEG_ftFolder '/' sprintf('sj%02d_exerCond%02d_changeDect_wmTF_bl01_ft.set',sjNum, exerCon)]);
    EEG = pop_loadset('filename',d(1).name,'filepath',EEG_ftFolder);
    
    % epoch files (first pass, epoch around fixation point because we
    % know that will be present in all trials, event broken ones)
    EEG = pop_epoch(EEG,{102},[-.5 4]);
    
    % diagnore missing triggers (only activate if you run into issues)
    %         EEG= pop_epoch(EEG,{201 201 202 203 204 205 206 211 212 213 214 215 216},[0 .01])
    %         EEG = pop_rmbase(EEG,[]); % remove baseline from whole epoch
    
    %this segment prevents multiple target triggers being coded into an epoch
    %in the event of a broken trial (which messes up the later epoching around the targets)
    for i=1:length(EEG.epoch)
        
        %if merged file then events are strings, if not merged they are
        %numbers...
        if mergedFile==1
            %output vector indicating target stims in epoch
            tmpVec =ismember( EEG.epoch(i).eventtype,{'201','202','203','204','205','206','207','208'});
        elseif mergedFile==0
            %output vector indicating target stims in epoch
            tmpVec = ismember(cell2mat(EEG.epoch(i).eventtype),[201,202,203,204,205,206,207,208]);
        end
        
        %if tmpVec indicates more than one target code in epoch
        if sum(tmpVec)>1
            thisCode = EEG.epoch(i).eventtype(tmpVec);
            EEG.epoch(i).eventtype = thisCode(1); %broken trial to scrap all triggers except the first single target one
        end
    end
    i = []; tmpVec = []; thisCode = [];
    
    % subject exception 
    %sj 4 had to quit before completing experiment
    if sjNum==4
        EEG = pop_select(EEG,'trial',1:424);
    end
    
    % load behavioral data from matlab (.mat file, must be merged)
    trialMat = [];
    trialMat = load([behMergedFolder '/' sprintf('sj%02d_exerCond%02d_changeDect_all_beh.mat',sjNum, exerCon)]);
    
    % subject exception
    if sjNum==15 && exerCon == 1
        trialMat.allTrialData(518:596) = [];
    elseif sjNum == 23 && exerCon == 1
        trialMat.allTrialData(309:316) = [];
    end
    
    % check for EEG/MAT consistency, break script if mismatched (note:
    % can only activate "break" if not running parfor)
    if length(trialMat.allTrialData) == length(EEG.epoch)
        disp('EEG AND MAT FILES MATCHING LENGTH!!!')
    else
        disp('MISMATCH BETWEEN EEG AND MAT FILES!!!')
        %break
    end
    
    % add trialinfo structure to the EEG
    EEG.trialInfo = trialMat.allTrialData;
    trialMat.allTrialData = [];
    
    % idenify broken trials (create vector)
    brkCounter=0;
    for i=1:length(EEG.trialInfo)
        if EEG.trialInfo(i).brokenTrial == 0
            brkCounter=brkCounter+1;
            brkTrialVector(brkCounter) = i;
            EEG.newTrialInfo(brkCounter) = EEG.trialInfo(i);
        end
    end
    
    % clears old EEG.trialInfo to avoid confusion at later stages!
    EEG.trialInfo = [];
    
    % selects unbroken trials only
    EEG = pop_select(EEG,'trial',brkTrialVector);
    
    % subject exception
    if sjNum==1601
        EEG = pop_select(EEG,'trial',1:957);
        EEG.newTrialInfo(958) = [];
    end
    
    % check EEG.epoch and EEG.newTrialInfo contain same number of
    % trials (another consistency check)
    if length(EEG.epoch)==length(EEG.newTrialInfo)
        disp(['EEG NOW CONTAINS ' num2str(length(EEG.epoch)) ' UNBROKEN TRIALS'])
    else
        disp('EEG.epoch or EEG.newTrialInfo DOES NOT CONTAIN CORRECT NO. TRIALS!  ABORT!')
        %break
    end
    
    % re-epoch data to [-.5 to 2.5] around target onset
    % CHANGED TO -1 to 2.5
    EEG = pop_epoch(EEG,{201 202 203 204 205 206 207 208},[-1 2.5]);
    
    % subject exception
    if sjNum==303
        EEG.newTrialInfo(265) =[];
    end
    
    % use this to diagnose mismatching data (occasional dropped epoch
    % code will result in EEG/BEH files not synching properly)
    findMismatchMat = [];
    %if ismember(sjNum,[101,303,603,604,801,1501,1904,1901,2002,2003,2103])   % if codes are strings
    for i=1:size(EEG.epoch,2)
        %findMismatchMat(i,:) = [str2num(EEG.epoch(i).eventtype{1}), EEG.newTrialInfo(i).locTrigger];
        eeg_locTrigger = str2num(EEG.epoch(i).eventtype{dsearchn([cellfun(@str2num,EEG.epoch(i).eventtype)]',200)});
        findMismatchMat(i,:) = [eeg_locTrigger, EEG.newTrialInfo(i).locTrigger];
    end
    % if 255 is first event, use second event code for comparison
    if findMismatchMat(i,1)==255|| findMismatchMat(i,1)==223
        findMismatchMat(i,:) = [EEG.epoch(i).eventtype{2}, EEG.newTrialInfo(i).locTrigger];
    end
    %         else % if event codes are numerical
    %             for i=1:size(EEG.epoch,2)
    %                 findMismatchMat(i,:) = [EEG.epoch(i).eventtype{1}, EEG.newTrialInfo(i).locTrigger];
    %                 % if 255 is first event, use second event code for comparison
    %                 if findMismatchMat(i,1)==255|| findMismatchMat(i,1)==223
    %                     findMismatchMat(i,:) = [EEG.epoch(i).eventtype{2}, EEG.newTrialInfo(i).locTrigger];
    %                 end
    %             end
    %%nd
    
    % gen. a third column to check epoch and trial values
    for i=1:size(EEG.epoch,2)
        findMismatchMat(i,3) = findMismatchMat(i,1) - findMismatchMat(i,2);
    end
    
    if sum(findMismatchMat(:,3))==0
        thisMatch=1;
        disp('EPOCH CODES CONSISTENT WITH MAT CODES!!!');
    else
        thisMatch=0;
        disp(['EPOCH CODES INCONSISTENT WITH MAT CODES FOR SJ ' num2str(sjNum)  ' -ABORT!!!'])
        %break
    end
    
    % create a mismatch log
    misMatchLog(sjNum,:) = [sjNum, thisMatch];
    
    % remove baseline (from whole epoch/baseline - see earlier setting)
    EEG = pop_rmbase(EEG,thisBaseline); % remove baseline from whole epoch
    
%     % do CRLS artifact correction (regresses out eye-closure shifts that
%     % would otherwise cause issues with later threshold based art. rej )
%     if artifactCorrection==1
%         EEG = pop_crls_regression( EEG, [67:72], 1, 0.9999, 0.01,[]);
%     end
    
    %save the synched, epoched data
    EEG = pop_saveset(EEG,'filename',sprintf('%s_ep.set',d(1).name(1:end-4)),'filepath',EEG_ft_epFolder);
    
    %save trial data for behavioral analysis only (this excludes
    %broken trials)
    trialInfoUnbroken = EEG.newTrialInfo;
    
    cd ('/home/garrett')
    parsave([unbrokenTrialsFolder '/' sprintf('sj%02d_exerCond%02d_changeDect_newBeh.mat',sjNum, exerCon)],trialInfoUnbroken)
    trialInfoUnbroken = [];
    
end


% save mismatch checker
%save('misMatchLog.mat','misMatchLog')

% close matlab pool
%matlabpool close

% run a function to find the fewest trials across all four conditions for
% each subject (only run if I also run doHilbert)
if doHilbert==1
    findFewestTrials % saves a mat called "Minimum_Location_Bin_Mat.mat" in the main dir
end

clear all
close all
end