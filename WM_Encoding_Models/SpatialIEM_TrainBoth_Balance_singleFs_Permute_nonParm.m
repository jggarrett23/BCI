%==========================================================================



function SpatialIEM_TrainBoth_Balance_singleFs_Permute_nonParm (subject)

close all

% which subjects?
%subject = [4];

%which condition?
%cond = [1];

em.nPerms =10; % how many permutations? %%%TOM EDIT THIS WAS SET TO 10!!%%%
nPerms = em.nPerms;

%name = '_SpatialTF.mat'; % name of files to be saved

% % setup directories
% root = pwd; out = 'AnalysisScripts/Cluster';
% dRoot = [root(1:end-length(out)),'Data/'];
% eRoot = [root(1:end-length(out)),'EEG/'];
% %bRoot = [root(1:end-length(out)),'Behavior/'];

% setup directories
root = '/home/garrett/WTF_Bike';
dRoot = [root '/Data/TrainBoth/'];
eRoot = [root '/EEG/'];



name = '_SpatialTF_allFreq.mat';  % name of files to be saved
allFreqs = 4:30;

% open matlab pool
% matlabpool open 72

% Loop through participants
fprintf('Subject:\t%02d\n',subject)

% designate em structure at beginning for PARFO
em = [];
EEG = [];

% Grab TF data file
fName = [dRoot, sprintf('sj%02d_TrainBoth_changeDect_accTrials',subject), name];
thisFile = load(fName);
em = thisFile.em;
thisFile = [];

%TOM ADDED
em.time = -.5*1000:1000/em.Fs:1.9961*1000; %   -500:4:2000; % time points of interest

% get analysis settings from TF data file.
nChans = em.nChans;
nBins = em.nBins;
nIter = 1; %em.nIter;
%nIter=10;
nBlocks = em.nBlocks;
freqs = allFreqs;
times = em.time;
nFreqs = length(allFreqs)-1;
%%nElectrodes = em.nElectrodes;
nSamps = length(em.time);
Fs = em.Fs;
basisSet = em.basisSet;
posBin = em.posBin;
nTrialsPerBlock = em.nTrialsPerBlock;

% Grab data------------------------------------------------------------

% Get EEG data
restfName = [eRoot, sprintf('sj%02d_exerCon01_changeDect_EEG_accTrials.mat',subject)];
exfName = [eRoot, sprintf('sj%02d_exerCon02_changeDect_EEG_accTrials.mat',subject)];

rest_tmp = []; rest_eeg = [];
rest_tmp = load(restfName);
rest_eeg = rest_tmp.eeg;

ex_tmp = []; ex_eeg = [];
ex_tmp = load(exfName);
ex_eeg = ex_tmp.eeg;

%Jordan Added
eeg.data = cat(1,rest_eeg.data,ex_eeg.data); %merge eeg trials along first dimension (trials x channels x timepoints)
eeg.arf.artIndCleaned = [rest_eeg.arf.artIndCleaned, ex_eeg.arf.artIndCleaned];

eeg.preTime = rest_eeg.preTime;
eeg.postTime = rest_eeg.postTime;
eeg.chanLabels = rest_eeg.chanLabels;
eeg.sampRate = rest_eeg.sampRate;

% get n channels (to save later with TF files)
%%% nElects = size(eeg.chanLabels,2);

eegs = eeg.data(:,:,:); % get scalp EEG (drop EOG electrodes)
artInd = eeg.arf.artIndCleaned.'; % grab artifact rejection index
tois = ismember(eeg.preTime:1000/Fs:eeg.postTime,em.time); nTimes = length(tois); % index time points for analysis.

% %     %% TOM EDIT TO PROCESS WHOLE EPOCH (WTF DATA EPOCHED FROM -.5 to 2)
% %     tois = ones(1,size(eegs,3)); nTimes = length(tois);

trial_indx = [repmat(1,1,length(rest_eeg.newTrialInfo)), repmat(2,1,length(ex_eeg.newTrialInfo))];
trial_indx = trial_indx(~artInd);
trial_indx = trial_indx';


% Remove rejected trials
eegs = eegs(~artInd,:,:);
posBin = posBin(~artInd);

em.nTrials = length(posBin); nTrials = em.nTrials; % # of good trials

%----------------------------------------------------------------------

% Preallocate Matrices
%tf_evoked = nan(nFreqs,nIter,nPerms,nSamps,nChans); 
tf_total_rest = nan(nFreqs,nIter,nPerms,nSamps,nChans);
tf_total_ex = tf_total_rest;
%C2_evoked = nan(nFreqs,nIter,nPerms,nSamps,nBins,nChans); 
C2_total_rest = nan(nFreqs,nIter,nPerms,nSamps,nBins,nChans);
C2_total_ex = C2_total_rest;

permInd = nan(nFreqs,nIter,nPerms,nBlocks,nTrialsPerBlock);
permedBins = nan(1,nTrialsPerBlock);

% TOM ADDED (needed to convert to EEGLAB format to use new eeglab filter)
EEG.data = permute(eegs,[2,3,1]); % converts to chans x times x trials
EEG.srate = Fs;
EEG.trials = size(EEG.data,3);
EEG.nbchan = size(EEG.data,1);
EEG.pnts = size(EEG.data,2);
data = double(EEG.data);

% Loop through each frequency
for f = 1:nFreqs
    fprintf('Frequency Band %d out of %d\n', f, nFreqs)
    
    f1 = allFreqs(f);
    f2 = allFreqs(f+1);
    
    % get no. of electrodes
    nElectrodes = size(eeg.data,2);
    disp(['nElecrodes changed to :' num2str(nElectrodes)])
    
    % BUTTERWORTH FILTER
    filterorder = 3;
    type = 'bandpass';
    [z1,p1] = butter(filterorder, [f1, f2]./(EEG.srate/2),type);
    %freqz(z1,p1,[],250)
    tempEEG = NaN(size(data,1),EEG.pnts,size(data,3));
    for x = 1:size(data,1)
        for y = 1:size(data,3)
            dataFilt1 = filtfilt(z1,p1,data(x,:,y)); % was filtfilt
            tempEEG(x,:,y) = dataFilt1; % tymp = chans x times x trials
        end
    end
    
    %% apply hilbert to each channel and epoch in turn (this should be correct)
    eegs = [];
    for j=1:size(tempEEG,1) % chan loop
        for i=1:size(tempEEG,3) % trial loop
            eegs(i,j,:) = hilbert(squeeze(tempEEG(j,:,i)));
        end
    end
    
    % eegs is trials x elects x times
    %fdata_evoked = eegs;
    fdata_total = abs(eegs).^2;
    
    
    % Loop through each iteration
    for iter = 1:nIter
        
        fprintf('Iteration %d out of %d\n', iter, nIter)
        
        blocks = em.blocks(:,iter); % grab blocks assignment for current iteration
        
        % Loop through permutations
        
        for perm = 1:nPerms
            tic % start timing permutation loop
            %fprintf('Permutation %d out of %d\n',perm,nPerms);
            
            %-----------------------------------------------------------------------------
            % Permute trial assignment within each block
            %-----------------------------------------------------------------------------
            permedPosBin = nan(size(posBin)); % preallocate permuted position bins vector
            
            perm_Trials = nan(size(trial_indx));
            pInd_Trial = [];
            for b = 1:nBlocks % for each block..
                pInd = randperm(nTrialsPerBlock); % create a permutation index
                permedBins(pInd) = posBin(blocks == b); % grab block b data and permute according data according to the index
                permedPosBin(blocks == b) = permedBins; % put permuted data into permedPosBin
                permInd(f,iter,perm,b,:) = pInd; % save the permutation (permInd is saved at end of the script)
            
                pInd_Trial(pInd) = trial_indx(blocks == b);
                perm_Trials(blocks == b) = pInd_Trial;
            end
            
            %-----------------------------------------------------------------------------
            
            % Average data for each position bin across blocks
            posBins = 1:nBins;
            blockDat_evoked = nan(nBins*nBlocks,nElectrodes,nSamps); % averaged evoked data
            blockDat_total = nan(nBins*nBlocks,nElectrodes,nSamps);  % averaged total data
            labels = nan(nBins*nBlocks,1);                           % bin labels for averaged data
            blockNum = nan(nBins*nBlocks,1);                         % block numbers for averaged data
            c = nan(nBins*nBlocks,nChans);                           % predicted channel responses for averaged data
            bCnt = 1;
            
            %blockDat_evoked_Rest = nan(nBins*nBlocks/2,nElectrodes,nSamps);
            %blockDat_evoked_Low = blockDat_evoked_Rest;
            blockDat_total_Rest = nan(nBins*nBlocks/2,nElectrodes,nSamps); 
            blockDat_total_Low = blockDat_total_Rest;
            
            rest_labels = nan(nBins*nBlocks/2,1);
            ex_labels = rest_labels;
            
            rest_blockNum = nan(size(rest_labels));
            ex_blockNum = rest_blockNum;
            
            rest_c = nan(nBins*nBlocks/2,nChans);
            ex_c = rest_c;
            
            rest_bCnt = 1; ex_bCnt = 1;
            
            
            for ii = 1:nBins
                for iii = 1:nBlocks
                    %blockDat_evoked(bCnt,:,:) = abs(squeeze(mean(fdata_evoked(permedPosBin==posBins(ii) & blocks==iii,:,tois),1))).^2;
                    blockDat_total(bCnt,:,:) = squeeze(mean(fdata_total(permedPosBin==posBins(ii) & blocks==iii,:,tois),1));
                    labels(bCnt) = ii;
                    blockNum(bCnt) = iii;
                    c(bCnt,:) = basisSet(ii,:);
                    
                    if iii == 1 || iii == 2
                        %blockDat_evoked_Rest(rest_bCnt,:,:) = abs(squeeze(mean(fdata_evoked(permedPosBin==posBins(ii) & blocks==iii & perm_Trials == 1,:,tois),1))).^2;
                        blockDat_total_Rest(rest_bCnt,:,:) = squeeze(mean(fdata_total(permedPosBin==posBins(ii) & blocks==iii & perm_Trials == 1,:,tois),1));
                        
                        rest_labels(rest_bCnt,:) = ii;
                        rest_blockNum(rest_bCnt,:) = iii;
                        
                        rest_bCnt = rest_bCnt + 1;
                    elseif iii == 3 || iii == 4
                        %blockDat_evoked_Low(ex_bCnt,:,:) = abs(squeeze(mean(fdata_evoked(permedPosBin==posBins(ii) & blocks==iii & perm_Trials == 2,:,tois),1))).^2;
                        blockDat_total_Low(ex_bCnt,:,:) = squeeze(mean(fdata_total(permedPosBin==posBins(ii) & blocks==iii & perm_Trials == 2,:,tois),1));
                        
                        ex_labels(ex_bCnt,:) = ii;
                        ex_blockNum(ex_bCnt,:) = iii;
                        
                        ex_bCnt = ex_bCnt + 1;
                    end
                    
                    
                    
                    bCnt = bCnt+1;
                end
            end
            
            for t = 1:nSamps
                
                % grab data for timepoint t
                toi = ismember(times,times(t)-em.window/2:times(t)+em.window/2); % time window of interest
                de = squeeze(mean(blockDat_evoked(:,:,toi),3)); % evoked data
                dt = squeeze(mean(blockDat_total(:,:,toi),3));  % total data
                
                % Do forward model
                tmpeC2 = nan(nBlocks,nBins,nChans); tmptC2 = tmpeC2; % for unshifted channel responses
                tmpeCR = nan(nBlocks,nChans); tmptCR = nan(nBlocks,nChans); % for shifted channel respones
                
                %Jordan Added
                %de_rest = squeeze(mean(blockDat_evoked_Rest(:,:,toi),3));
                dt_rest = squeeze(mean(blockDat_total_Rest(:,:,toi),3));
                
                %de_low = squeeze(mean(blockDat_evoked_Low(:,:,toi),3));
                dt_low = squeeze(mean(blockDat_total_Low(:,:,toi),3));
                
                %tmpeCR_rest = nan(nBlocks/2,nChans); tmpeCR_low = tmpeCR_rest;
                tmptCR_rest = nan(nBlocks/2,nChans); 
                tmptCR_low = tmptCR_rest;
                
                %restEvoked_i = 1; 
                restTotal_i = 1;
                %exEvoked_i = 1; 
                exTotal_i = 1;
                
                
                for i=1:nBlocks % loop through blocks, holding each out as the test set
                    
                    trnl = labels(blockNum~=i); % training labels
                    tstl = labels(blockNum==i); % test labels
                    
                    rest_tstl = rest_labels(rest_blockNum == i);
                    ex_tstl = ex_labels(ex_blockNum == i);
                
                    %-----------------------------------------------------%
                    % Analysis on Evoked Power                            %
                    %-----------------------------------------------------%
%                     B1 = de(blockNum~=i,:);    % training data
%                     B2 = de(blockNum==i,:);    % test data
%                     C1 = c(blockNum~=i,:);     % predicted channel outputs for training data
%                     W = C1\B1;          % estimate weight matrix
%                     C2 = (W'\B2')';     % estimate channel responses
%                     
%                     % tmpeC2(i,:,:) = C2;
%                     
%                     % shift eegs to common center
%                     n2shift = ceil(size(C2,2)/2);
%                     for ii=1:size(C2,1)
%                         [~, shiftInd] = min(abs(posBins-tstl(ii)));
%                         C2(ii,:) = wshift('1D', C2(ii,:), shiftInd-n2shift-1);
%                     end
%                     
%                     tmpeCR(i,:) = mean(C2); % average shifted channel responses
                    
                    %-----------------------------------------------------%
                    % Analysis on Total Power                             %
                    %-----------------------------------------------------%
                    B1 = dt(blockNum~=i,:);    % training data
                    B2 = dt(blockNum==i,:);    % test data
                    C1 = c(blockNum~=i,:);     % predicted channel outputs for training data
                    W = C1\B1;          % estimate weight matrix
                    C2 = (W'\B2')';     % estimate channel responses
                    
                    % tmptC2(i,:,:) = C2;
                    
                    % shift eegs to common center
                    n2shift = ceil(size(C2,2)/2);
                    for ii=1:size(C2,1)
                        [~, shiftInd] = min(abs(posBins-tstl(ii)));
                        C2(ii,:) = wshift('1D', C2(ii,:), shiftInd-n2shift-1);
                    end
                    
                    tmptCR(i,:) = mean(C2); % averaged shifted channel responses
                    
                    if i == 1 || i == 2
                        B2_rest = dt_rest(rest_blockNum==i,:);
                        C2_rest = (W'\B2_rest')'; % test
                        C2_total_rest(f,iter,t,restTotal_i,:,:) = C2_rest;
                        
                        %shift conditions
                        n2shift = ceil(size(C2_rest,2)/2);
                        for ii=1:size(C2_rest,1)
                            [~, shiftInd] = min(abs(posBins-rest_tstl(ii)));
                            C2_rest(ii,:) = wshift('1D', C2_rest(ii,:), shiftInd-n2shift-1);
                        end
                        
                        tmptCR_rest(restTotal_i,:) = mean(C2_rest);
                        
                        restTotal_i = restTotal_i + 1;
                    elseif i == 3 || i == 4
                        
                        B2_low = dt_low(ex_blockNum==i,:);
                        C2_low = (W'\B2_low')'; % test
                        C2_total_low(f,iter,t,exTotal_i,:,:) = C2_low;
                        
                        
                        n2shift = ceil(size(C2_low,2)/2);
                        for ii=1:size(C2_low,1)
                            [~, shiftInd] = min(abs(posBins-ex_tstl(ii)));
                            C2_low(ii,:) = wshift('1D', C2_low(ii,:), shiftInd-n2shift-1);
                        end
                        
                        tmptCR_low(exTotal_i,:) = mean(C2_low);
                        
                        exTotal_i = exTotal_i + 1;
                    end
                    
                    
                    %-----------------------------------------------------%
                    
                end
                % save data to indexed matrix
                % C2_evoked(f,iter,perm,t,:,:) = mean(tmpeC2);
                % C2_total(f,iter,perm,t,:,:) = mean(tmptC2);
                %tf_evoked(f,iter,perm,t,:) = mean(tmpeCR);
                tf_total(f,iter,perm,t,:) = mean(tmptCR);
                
                %tf_evoked_rest(f,iter,perm,t,:) = mean(tmpeCR_rest);
                tf_total_rest(f,iter,perm,t,:) = mean(tmptCR_rest);
                
                %tf_evoked_low(f,iter,perm,t,:) = mean(tmpeCR_low);
                tf_total_low(f,iter,perm,t,:) = mean(tmptCR_low);
                
            end
            toc
        end
    end
    toc % stop timing the frequency loop
end

%% useful code for dealing with many iterations (v.large mats)
% %         % average over 1000 ITS + 3 BLOCKS to reduce size of saved file!
% %         tf_evoked = squeeze(mean(tf_evoked,4));
% %         tf_total = squeeze(mean(tf_total,4));

%         tf_evoked = squeeze(mean(mean(tf_evoked,1),2)); % average across freqs (only 1) and iterations - left with perm x times x chans mat
%         tf_total = squeeze(mean(mean(tf_total,1),2));

fName = [dRoot,sprintf('sj%02d_TrainBoth_Permute_changeDect_accTrials_',subject),name];
em.permInd = permInd;
%em.permtfs.evoked = tf_evoked;
em.permtfs.total = tf_total;
em.permtfs.tfs_rest.total = tf_total_rest;
em.permtfs.tfs_low.total = tf_total_low;

cd ([root, '/Analysis_Scripts'])
save(fName, 'em', 'nElectrodes','-v7.3');

    

    
%sendEmailToMe('SPATIAL IEM PERMUTE SCRIPT FINISHED PROCESSING!!')
    