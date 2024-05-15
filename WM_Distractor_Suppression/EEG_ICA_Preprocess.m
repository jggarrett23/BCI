function EEG_ICA_Preprocess(sjNum, condition, mergeData, filterData, trainICA,...
    applyICA, epochData)

cd /home/garrett/eeglab14_1_2b
eeglab;
close all
cd /work/garrett/CDA_Bike/Analysis_Scripts


% seed rng
for rngShuffle =1:100
    rng('shuffle')
end

parent_folder = '/work/garrett/CDA_Bike/';
rawDir = [parent_folder, 'EEG_Raw/'];
mergeDir = [parent_folder,'EEG_Merged/'];
ftDir = [parent_folder 'ICA_Preprocess/Filt'];
amicaDir = [parent_folder 'ICA_Preprocess/Amica_Output'];
unbrokenTrialsFolder = [parent_folder 'Beh_Unbroken'];
epDir = [parent_folder 'ICA_Preprocess/Epoch/'];
behDir = [parent_folder 'Beh_merged/'];

thisBaseline = [-200 0]; % Based off Adam, Robinson, & Vogel, 2018
 
% create some blank variables (these are for synch consistency checks)
misMatchLog = [];

brkTrialVector=[]; EEG = [];                                            % 

%% set subject numbers and define exceptions (e.g. split EEG files due to bathroom breaks)
disp(['PROCESSING SUBJECT: ' num2str(sjNum)])

if mergeData
    
    %cd into the raw data (edf) directorty
    cd(rawDir)
    
    % get total number of eeg files
    
    %naming messed up for subject 2 condition 1
    all_eegFiles=dir(sprintf('Expsj%02d_cond%02d_bl*.vhdr',sjNum, condition));
    
    nBlocks = length(all_eegFiles);
    
    if sjNum == 32 && condition == 1
        nBlocks = nBlocks - 1;
    end
    
    if sjNum == 11 && condition == 2 % did not finish study
        total_blocks = 9;
    else
        total_blocks = 10;
    end
    
    % incase of duplicate files or not enough blocks
    if nBlocks ~= total_blocks
        error(sprintf('Check number of blocks for Sj %02d Con %02d', sjNum, condition));
        return
    end
    
    % merge blocks into one file
    for iBlock = 1:nBlocks
        
        
        file_prefix = sprintf('Expsj%02d_cond%02d_bl%02d',sjNum,condition,iBlock);
        
        file_bool = cellfun(@(x) startsWith(x,file_prefix), {all_eegFiles.name});
        
        if sjNum == 32 && condition == 1 && iBlock == 8
           % battery died in middle of block 8, need to concatenate those files first
           file1 = pop_fileio('Expsj32_cond01_bl08_cdaBike_EEG.vhdr');
           file2 = pop_fileio('Expsj32_cond01_bl08_1.vhdr');
           currentBlock_EEG = pop_mergeset(file1, file2);
        else
           filename = all_eegFiles(file_bool).name;
           currentBlock_EEG = pop_fileio(filename);
        end
    
        if iBlock == 1
            EEG = currentBlock_EEG;
        else
            EEG = pop_mergeset(EEG,currentBlock_EEG);
        end
        
    end
    
    % save filtered data
    EEG = pop_saveset(EEG,'filename',sprintf('sj%02d_cond%02d_wmFilt.set',sjNum, condition),'filepath',mergeDir);
    
end


%% import/filter data routine
if filterData
   
    EEG = pop_loadset('filename',sprintf('sj%02d_cond%02d_wmFilt.set',sjNum, condition), 'filepath',mergeDir);
    
    % add channel locations
    EEG=pop_chanedit(EEG, 'lookup','/home/garrett/eeglab14_1_2b/plugins/dipfit3.2/standard_BESA/standard-10-5-cap385.elp'); % cluster
    
    EEG.data = double(EEG.data);
    
    %remove HEOG and AUX channels
    EEG = pop_select(EEG,'nochannel',{'HEOG', 'LEG', 'HEAD'});
    
    % store raw unprocessed data to project extracted components back onto
    EEG_2 = EEG;
    
    % high pass filter at 1 Hz
    EEG = pop_eegfiltnew(EEG,1);
    
    % apply 0.1-30 Hz filter to EEG 2
    EEG_2 = pop_eegfiltnew(EEG_2, .1, 30);
    
    % resample after before filtering to save time (arguments for after
    % filtering to avoid anti-aliasing)
    EEG = pop_resample(EEG,250);
    
    EEG_2 = pop_resample(EEG_2, 250);
    
    % Remove bad electrodes before average reference
    EEG_clean = clean_artifacts(EEG, 'Highpass','off','BurstCriterion','off',...
        'WindowCriterion','off','BurstRejection','off','Distance','Euclidean');
        
    % Get bad channel names
    clean_chans = {EEG_clean.chanlocs.labels};
    all_chans = {EEG.chanlocs.labels};
    
    bad_chans = setdiff(all_chans, clean_chans);
    bad_chansIdx = cellfun(@(i) find(strcmp(i,all_chans)), bad_chans);
    
    % Interpolate bad channels
    EEG = pop_interp(EEG, bad_chansIdx, 'spherical');
    
    EEG_2 = pop_interp(EEG_2, bad_chansIdx, 'spherical');
    
    % store for interpolating raw data
    EEG.bad_chansIdx = bad_chansIdx;
    
    % clear up space
    clear EEG_clean
    
    % average reference
    EEG = pop_reref(EEG, [],'keepref','on');
    
    EEG_2 = pop_reref(EEG_2, [], 'keepref','on');
    
    EEG.processed.filt_range = [0.1, 30];
    EEG.processed.data = EEG_2.data;
    
    % save filtered data
    EEG = pop_saveset(EEG,'filename',sprintf('sj%02d_cond%02d_wmFilt_ft.set', sjNum, condition),'filepath',ftDir);
    
    %cd back to main directory
    cd(parent_folder)
    
end

if trainICA
    
    EEG = pop_loadset([ftDir, sprintf('/sj%02d_cond%02d_wmFilt_ft.set', sjNum, condition)]);
    
    % Apply AMICA to filtered and channel rejected data
    numprocs = 1; % number of nodes
    num_models = 1; % number of models of mixture ICA
    max_threads = 1; % number of threads
    max_iters = 2000; % number of learning steps default=2000
    
    dataRank = sum(eig(cov(EEG.data')) > 1E-6);
    
    outfile = [amicaDir sprintf('/sj%02d_cond%02d_wmFilt_amica', sjNum, condition)];
    
    [weights, sphere, mods] = runamica15(EEG.data, 'num_models', num_models,...
        'pcakeep', dataRank, 'numprocs', numprocs, 'max_iter', max_iters,...
        'max_threads', max_threads, 'do_reject',1, 'numrej',15, 'rejsig', 3, 'rejint', 1,...
        'outdir',outfile);
end

if applyICA
    
    EEG = pop_loadset([ftDir, sprintf('/sj%02d_cond%02d_wmFilt_ft.set', sjNum, condition)]);
    
    amica_outfile = [amicaDir sprintf('/sj%02d_cond%02d_wmFilt_amica', sjNum, condition)];
    
    EEG = pop_loadmodout(EEG, amica_outfile);
    EEG = eeg_checkset(EEG);
    
    % Label Components
    EEG = pop_iclabel(EEG, 'default');
    
    % classifications stored in etc
    ic_classifications = EEG.etc.ic_classification.ICLabel;
    
    ic_classLabels = ic_classifications.classes;
    
    brain_comp_thresh = 0.8;
    
    % keep only brain components over 90% classification probability
    brainLabel_component_idx = find(strcmp('Brain',ic_classLabels));
    
    brain_comp_probs = ic_classifications.classifications(:,brainLabel_component_idx);
    
    [comp_maxClass_probs, comp_maxClass_labels] = max(ic_classifications.classifications, [],2);
    
    EEG.brain_compsIdx  = brain_comp_probs >= brain_comp_thresh;
    
    EEG.keep_comps = intersect(find(EEG.brain_compsIdx), find(comp_maxClass_labels == 1));
    
    if isempty(EEG.keep_comps)
        warning(sprintf('No components identified as being brain activity for Sj %02d Condition %02d', sjNum, condition))
    end
    
    % Project components onto raw data
    EEG.filt_data = EEG.data; % store incase
    
    % Project brain components onto data
    EEG.data = EEG.processed.data;
    
    % keep only brain components
    EEG = pop_subcomp(EEG, EEG.keep_comps, 0, 1);
    
    EEG = eeg_checkset(EEG);
    
    EEG.etc.ic_classification.ICLabel.classifications = ic_classifications.classifications(EEG.keep_comps,:);
    
    EEG.icaact = [];
    
    EEG = pop_saveset(EEG,'filename',sprintf('sj%02d_cond%02d_wmFilt_ft_ICA.set', sjNum, condition),'filepath',amicaDir);
    
end


if epochData
    
    EEG = pop_loadset([amicaDir, sprintf('/sj%02d_cond%02d_wmFilt_ft_ICA.set', sjNum, condition)]);
    
    % epoch files (first pass, epoch around cue presentation point because we
    % know that will be present in all trials, even broken ones)
    EEG = pop_epoch(EEG,{'S102'},[-.5 1.5]);
    
    %this segment prevents multiple target triggers being coded into an epoch
    %in the event of a broken trial (which messes up the later epoching around the targets)
    for i=1:length(EEG.epoch)
        
        tmpVec =ismember( EEG.epoch(i).eventtype,{'S201','S203','S204','S211','S213','S214'});
        
        %if tmpVec indicates more than one target code in epoch
        if sum(tmpVec)>1
            thisCode = EEG.epoch(i).eventtype(tmpVec);
            EEG.epoch(i).eventtype = thisCode(1); %broken trial to scrap all triggers except the first single target one
        end
    end
    i = []; tmpVec = []; thisCode = [];
    
    % load behavioral data from matlab (.mat file, must be merged)
    load([behDir, sprintf('sj%02d_con%02d_allBeh.mat',sjNum, condition)], 'allTrialInfo');
   
    % check for EEG/MAT consistency, break script if mismatched (note:
    % can only activate "break" if not running parfor)
    if length(allTrialInfo) == length(EEG.epoch)
        disp('EEG AND MAT FILES MATCHING LENGTH!!!')
    else
        disp('MISMATCH BETWEEN EEG AND MAT FILES!!!')
        %break
    end
    
    % account for missed trials due to battery failure on block 8
    if sjNum == 32 && condition == 1
        
        % used below code to determine when mismatch occured, then
        % inspected when behavioral data and eeg data align. Mismatch
        % between 667-689, then [676:679, 682:684] 
        allTrialInfo([667:689, 699:705, 712:711]) = [];
        %allTrialInfo([676:679, 682:684]) = [];
        
        % Loop through each trial in eeg data and determine where
        % inconsistency occurs
       
        % EEG events
        eeg_events = {EEG.epoch.eventtype};
        
        % convert behavioral data set size labels to EEG labels for
        % comparison
        setSizes = [allTrialInfo.SetSize];
        distPresent = find([allTrialInfo.Distractor_Present] == 1);
        cueLeft = [allTrialInfo.cueDir] == 0;
        cueRight = [allTrialInfo.cueDir] == 1;
        
        brokensIdx = find(setSizes == 999);
        
        % dummy code distractor trials
        setSizes(distPresent) = 3;
        
        setSizes(cueRight) = setSizes(cueRight) + 210;
        setSizes(cueLeft) = setSizes(cueLeft) + 200;
        
        mismatch_trials = [];
        eeg_event_codes = [];
        for iTrial = 1:length(eeg_events)
           
            beh_code = setSizes(iTrial);
            
            events = eeg_events{iTrial};
            event_codes = cellfun(@(x) str2double(erase(x, 'S')), eeg_events{iTrial});
            
            % need to figure out correct logic for broken trial mismatches
            if beh_code == 999 && and(~ismember(95, event_codes), any(event_codes >= 201))
                mismatch_trials = [mismatch_trials, iTrial];
            elseif beh_code ~= 999 && ~ismember(beh_code, event_codes)
                mismatch_trials = [mismatch_trials, iTrial];
            else
                
                if beh_code == 999
                    eeg_event_codes = [eeg_event_codes, 95];
                else
                    ev_code = event_codes(find(event_codes >= 201));
                    eeg_event_codes = [eeg_event_codes, ev_code];
                end
            end
        end
        
    
        unbroken_TrialVector = find(setSizes ~= 999);
        
    else
        
        % idenify broken trials (create vector)
        unbroken_TrialVector = find([allTrialInfo.SetSize] ~= 999);
        
    end
    
    trialInfoUnbroken = allTrialInfo(unbroken_TrialVector);
    EEG.newTrialInfo = trialInfoUnbroken;
    
    % save for behavioral analyses
    save([unbrokenTrialsFolder '/' sprintf('sj%02d_cond%02d_wmFilt_newBeh.mat',sjNum, condition)],'trialInfoUnbroken')
    
    clear trialInfoUnbroken
    
    % selects unbroken trials only
    EEG = pop_select(EEG,'trial',unbroken_TrialVector);
    
    % check EEG.epoch and EEG.newTrialInfo contain same number of
    % trials (another consistency check)
    if length(EEG.epoch)==length(EEG.newTrialInfo)
        disp(['EEG NOW CONTAINS ' num2str(length(EEG.epoch)) ' UNBROKEN TRIALS'])
    else
        disp('EEG.epoch or EEG.newTrialInfo DOES NOT CONTAIN CORRECT NO. TRIALS!  ABORT!')
        %break
    end
    
    % re-epoch data to [-.3 to 1.2] around target onset
    EEG = pop_epoch(EEG,{'S201','S203','S204','S211','S213','S214'},[-.5 1.1]);
    
    % use this to diagnose mismatching data (occasional dropped epoch
    % code will result in EEG/BEH files not synching properly)
    findMismatchMat = [];
    for i=1:size(EEG.epoch,2)
        
        stimTrigger_valIdx = logical(cellfun(@(x) sum(strcmp(x,{'S201','S203','S204','S211','S213','S214'})),EEG.epoch(i).eventtype));
        
        eegTrigger = regexprep(EEG.epoch(i).eventtype{stimTrigger_valIdx},'\D','');
        
        behTrial = EEG.newTrialInfo(i);
        if behTrial.Distractor_Present
            behTrigger = sprintf('2%d3',behTrial.cueDir);
        else
            behTrigger = sprintf('2%d%d',behTrial.cueDir,behTrial.SetSize);
        end
        
        findMismatchMat(i,:) = [str2num(eegTrigger), str2num(behTrigger)];
    end
    
    % gen. a third column to check epoch and trial values
    for i=1:size(EEG.epoch,2)
        findMismatchMat(i,3) = findMismatchMat(i,1) - findMismatchMat(i,2);
    end
    
    if sum(findMismatchMat(:,3))== 0
        thisMatch=1;
        disp('EPOCH CODES CONSISTENT WITH MAT CODES!!!');
    else
        thisMatch=0;
        error(['EPOCH CODES INCONSISTENT WITH MAT CODES FOR SJ ' num2str(sjNum)  ' -ABORT!!!'])
        %break
    end
    
    % create a mismatch log
    misMatchLog(sjNum,:) = [sjNum, thisMatch];
    
    % remove baseline (from whole epoch/baseline - see earlier setting)
    EEG = pop_rmbase(EEG,thisBaseline); % remove baseline from whole epoch
    
    EEG.accuracy = sum([EEG.newTrialInfo.Change_Type] == [EEG.newTrialInfo.Response]);
    
    % Extract only accurate trials
    EEG.correct_trials = [EEG.newTrialInfo.Change_Type] == [EEG.newTrialInfo.Response];
   
    EEG.data = EEG.data(:,:,EEG.correct_trials);
    EEG.newTrialInfo = EEG.newTrialInfo(EEG.correct_trials);
    
    save([epDir '/' sprintf('sj%02d_cond%02d_ICA_EEG.mat',sjNum, condition)],'EEG', '-v7.3');
    
end


end