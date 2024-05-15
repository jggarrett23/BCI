function classify_flickFreqs(sjNum)

%% Set-Up Directories
scriptDir = mfilename('fullpath');
parentDir = scriptDir(1:max(strfind(pwd,'\')));
dataDir = [parentDir 'ssvep\'];

sjNums = 1:2;
allSj_data = {};

load([dataDir 'classInfo_4_5.m'], 'classInfo_4_5');

for iSub = sjNums
    
    flickTrials_Idx = arrayfun(@(iCol) find(classInfo_4_5(:,iCol)), 1:size(classInfo_4_5,2), 'UniformOutput', false);
    flickTrials_Idx = cell2mat(flickTrials_Idx)'; % rows correspond to each flicker label
    trial_labels = zeros(numel(flickTrials_Idx),1);
    
    flick_freqs = [9, 10, 12, 15];
    nFlicks = length(flick_freqs);
    
    for iFlick = 1:nFlicks
        trial_labels(flickTrials_Idx(iFlick,:)) = iFlick;
    end
    
   % 11 x nTimepoints recording %(time points vary per file)
   % row 1: samp time
   % row 2-9: EEG
   % row 10: LED on/off trigger
   % row 11: LDA baseline classifications
   
   % Stimulation frequencies: 9, 10, 12 and 15 Hz
   
   % Notes: 
   % Data collected in Austria so line noise at 50 Hz
   % Original study bandpassed between 0.5-30Hz (data should be filtered in
   % same way)
   load([dataDir sprintf('subject_%d_fvep_led_training_1.mat', iSub)], 'y');
   
   eeg_data = double(y(2:9,:));
   time = y(1,:);
   baseline = y(11,:);
   triggers = y(10,:);
   nChans = size(eeg_data,1);
   srate = 256;
   
   % use on/off triggers to find contiguous seconds of stimulus flashing
   [epoch_idxs, timepoint_labels] = parse_data(triggers);
    
   
   % define time window parameters
    tWindow_s = 10; % size of time window in s
    wOverlap = 0.80; % amount of overlap between time windows

    tWindow_size = dsearchn(time',tWindow_s);
    
    [data_cropped, labels] = sliding_window_crops(eeg_data, timepoint_labels, time, tWindow_size, wOverlap);
    crop_time = time(1:tWindow_size);
    
    % set seed
    rng(23);
    
    nCrops = size(data_cropped,3);
    shuff_idx = randperm(nCrops);
    data_cropped = data_cropped(:,:,shuff_idx);
    labels = labels(:,shuff_idx);
    
    % loo-cv
    
    cv_results = [];
    
    
        
%         test_idx = iCrop;
%         train_idx = setdiff(1:nCrops, iCrop);
        
%         X_train = data_cropped(:,:,train_idx);
%         X_test = data_cropped(:,:,test_idx);
%         
%         % standardize_data
%         X_train_mu = mean(X_train,[1,3]);
%         X_train_sd = std(X_train, [], [1,3]);
%         X_train = (X_train - X_train_mu) ./ X_train_sd;
%         
%         X_test = (X_test - X_train_mu) ./ X_train_sd;
%         
%         y_train = labels(:,train_idx);
%         y_test = labels(:,test_idx);
%         
%         % use mode of labels as true label 
%         y_train_mode = mode(y_train,1)';
%         y_test_mode = mode(y_test);

        
        
% extract spectral features
X_phi = [];
parfor iCrop = 1:nCrops
    
    [Phi, omega, labmda, b, X_hat, S, mode_hz] = DMD(squeeze(data_cropped(:,:,iCrop)), crop_time*1000, 1/srate, 'doDelay', 1, 'delayEmbedding', 1000);
   
    P = diag(Phi'*Phi);
    flick_freqIdx = arrayfun(@(i) find(round(mode_hz) == i), flick_freqs, 'UniformOutput', false);
    
    flick_phi = cellfun(@(x) squeeze(mean(abs(Phi(:, x)), 2)), flick_freqIdx, 'UniformOutput', false);
    flick_phi = cell2mat(flick_phi);
    
    
    X_phi(iCrop,:, :) = flick_phi;
end

X_phi = reshape(X_phi, size(X_phi,1), size(X_phi,2)*size(X_phi,3));


mdl = fitcdiscr(X_phi, mode(labels,1)', 'SaveMemory', 'on', 'Leaveout', 'on');
test_loss = kfoldLoss(mdl);

 
foo = 0;
    
    
end

end

%% Helper Functions

function [epoch_idxs, y]= parse_data(triggers)

%{

    Trial structure:
    Each run began with a 10s delay. Next, each trial started with a 3s
    pause. Subjects were asked to fixate for 7s on flicker.

    % used a 200ms sliding window to classify stimulation frequencies

%}

% use on/off triggers to find contiguous seconds of stimulus flashing
trig_onIdx = find(triggers == 1);
trig_onDiffs = diff(trig_onIdx);
non_contiguousSegments = find(trig_onDiffs > 1);
trig_ends = trig_onIdx(non_contiguousSegments-1);

trig_offIdx = find(triggers == 0);
trig_offDiffs = diff(trig_offIdx);
non_contiguousSegments = find(trig_offDiffs > 1);
trig_starts = trig_offIdx(non_contiguousSegments);

epoch_idxs = [trig_starts, trig_ends] + 1;
epoch_idxs = sort(epoch_idxs);
epoch_idxs = [epoch_idxs trig_onIdx(end)];

nFlick_onTrials = length(epoch_idxs)/2; % should be 20
if nFlick_onTrials ~= 20
    error('Wrong number of Trials Extracted')
end

y = triggers;
label_cnt = 1;
for iTrial_starts = 1:2:length(epoch_idxs)
    trial_start = epoch_idxs(iTrial_starts);
    trial_end = epoch_idxs(iTrial_starts+1);
    if label_cnt > 4
        label_cnt = 1;
    end
    
    y(trial_start:trial_end) = label_cnt;
    label_cnt = label_cnt + 1;
end

end

function [data_cropped, y_cropped] = sliding_window_crops(data, y, time, tWindow_size, win_overlap)

tOverlap_step = floor(tWindow_size * (1-win_overlap));

if ~tOverlap_step
    tOverlap_step = 1;
end

tWindow_starts = 1:tOverlap_step:length(time)-tWindow_size;
nTwindows = length(tWindow_starts);
data_cropped = zeros(size(data,1), tWindow_size, nTwindows);
y_cropped = zeros(tWindow_size,nTwindows);
for iTwindow = 1:nTwindows
    tWindow_start = tWindow_starts(iTwindow);
    tWindow_end = tWindow_start + tWindow_size-1;
    
    data_cropped(:,:,iTwindow) = data(:,tWindow_start:tWindow_end);
    y_cropped(:,iTwindow) = y(tWindow_start:tWindow_end);
    
end


end

function cv = balanced_cv_split(labels,varargin)


p = inputParser;
validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);

addRequired(p, 'labels');
addOptional(p,'KFolds',5,validScalarPosNum); 
addOptional(p, 'TestSize', 0.1, validScalarPosNum);
addOptional(p, 'RandomState', 0);

parse(p,labels,varargin{:});

nFolds = p.Results.KFolds;
test_size = p.Results.TestSize;
random_state = p.Results.RandomState;

rng(random_state)


[counts, ~] = histcounts(labels, min(labels):max(labels)+1);
uniq_labels = unique(labels);

nTest_trials = ceil(min(counts)*test_size);

counts = counts - nTest_trials;

min_trials = min(counts) - 1;

label_idxs = [];
for iLabel = 1:length(uniq_labels)
    label_idxs(iLabel,:) = labels == uniq_labels(iLabel);
end  

train_idxs = {};
test_idxs = {};
for iFold = 1:nFolds

    allLabel_trainIdxs = [];
    allLabel_testIdxs = [];
    for iLabel_idx = 1:length(uniq_labels)

        current_labelIdx = find(label_idxs(iLabel_idx,:));
        
        shuffIdx = randperm(length(current_labelIdx));

        test_idx = current_labelIdx(shuffIdx(1:nTest_trials));

        train_idx = current_labelIdx(shuffIdx(nTest_trials+1:end));
        
        train_idx = train_idx(1:min_trials);

        allLabel_trainIdxs(iLabel_idx,:) = train_idx;
        allLabel_testIdxs(iLabel_idx,:) = test_idx;
       
    end
    
    train_idxs{iFold} = allLabel_trainIdxs(:);
    test_idxs{iFold} = allLabel_testIdxs(:);
    
end

cv.training = train_idxs;
cv.test = test_idxs;
cv.NumTestSets = nFolds;
       
end