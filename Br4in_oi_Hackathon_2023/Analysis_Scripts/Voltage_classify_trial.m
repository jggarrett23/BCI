function Voltage_classify_trial(sjNum)

%% Set Up
parentDir = '/work/garrett/DT_CDA/';
dataDir = [parentDir 'EEG/'];
saveDir = [parentDir 'Classifier_Results/'];


allData = {};
all_trialTypes = {};
all_cueDirs = {};
for iCon = [0,1]
    
    % load data
    load([dataDir sprintf('sj%02d_cond%02d_EEG.mat', sjNum, iCon)], 'EEG');

    % electrodes x time points x trials
    data = EEG.data;
    
    allData{iCon+1} = data;
    
    % extract specific trials
    ss1_trials = [EEG.newTrialInfo.SetSize] == 1;
    ss4_trials = intersect(find([EEG.newTrialInfo.SetSize] == 4), find([EEG.newTrialInfo.Distractor_Present] ~= 1));
    ss13_trials = intersect(find([EEG.newTrialInfo.SetSize] == 4), find([EEG.newTrialInfo.Distractor_Present]));
    
    y = zeros(length(ss1_trials),1); % ss1
    y(ss13_trials) = 1; % ss 1+3
    y(ss4_trials) = 2; % ss 4
    
    all_trialTypes{iCon+1} = y;
    
    cue_direction = [EEG.newTrialInfo.cueDir];
    all_cueDirs{iCon+1} = cue_direction;
    
end


time = EEG.times;
nChans = EEG.nbchan;
select_chans = 0;

chans_oi = {'PO7','O1','PO3','P3','P7','P5',...
    'PO8','O2','PO4','P4','P8','P6'};

chanlocs = EEG.chanlocs;

if select_chans
    chans_oi_idx = cellfun(@(x) find(strcmp(x,{chanlocs.labels})), chans_oi);
else 
    chans_oi_idx = 1:nChans;
end


%% Classify

rng(3);

singleTask_data = allData{1};
dualTask_data = allData{2};

singleTask_trialLabels = all_trialTypes{1};
dualTask_trialLabels = all_trialTypes{2};

singleTask_cueDirs = all_cueDirs{1};
dualTask_cueDirs = all_cueDirs{2};

% extract non-phase locked
singleTask_data = singleTask_data - mean(singleTask_data,3);
dualTask_data = dualTask_data - mean(dualTask_data,3);

X = cat(3, singleTask_data, dualTask_data);
y = [singleTask_trialLabels; dualTask_trialLabels];

X = X(chans_oi_idx, :,:);
X = double(X);

baseline = [-200, 0];

% baseline correct data
baseline_idx = dsearchn(time',baseline(1)):dsearchn(time',baseline(2));

X = X - mean(X(:, baseline_idx, :),2);

trial_types = {'1', '1+3', '4'};
y_trialTypes = arrayfun(@(x) trial_types{x+1}, y, 'UniformOutput', false);

cueDirs = [singleTask_cueDirs, dualTask_cueDirs]';

%train_test_split = cvpartition(y, 'Holdout', 0.1, 'Stratify', true);
train_test_split = balanced_cv_split(y, 'KFolds', 1, 'TestSize', 0.1, 'RandomState', 3);

%test_idx = test(train_test_split);
%train_val_idx = training(train_test_split);
test_idx = train_test_split.test{1};
train_val_idx = train_test_split.training{1};

X_test = X(:,:,test_idx);
y_test = y(test_idx);

cueDirs_test = cueDirs(test_idx);

X_train_val = X(:,:,train_val_idx);
y_train_val = y(train_val_idx);
y_train_valTrialTypes = y_trialTypes{train_val_idx};

cueDirs_train_val = cueDirs(train_val_idx);

nChans = size(X_test,1);

% define time window parameters
tWindow_ms = 10; % size of time window in ms
wOverlap = 0.80; % amount of overlap between time windows

tWindow_size = length(dsearchn(time', 0):dsearchn(time', tWindow_ms));

tOverlap_step = floor(tWindow_size * (1-wOverlap));

if ~tOverlap_step
    tOverlap_step = 1;
end

tWindow_starts = 1:tOverlap_step:length(time)-tWindow_size;

nTwindows = length(tWindow_starts);

nFolds = 10;
nCompares = 3;

time_val_loss = zeros(nCompares, nTwindows, nFolds);
time_val_specificity = zeros(nCompares, nTwindows, nFolds, length(trial_types));
time_train_loss = zeros(nCompares, nTwindows, nFolds);
time_train_specificy = zeros(nCompares, nTwindows, nFolds, nCompares-1);

train_mdl_weights = zeros(nCompares, nTwindows,nChans);

time_test_loss = zeros(nCompares, nTwindows);
time_test_specificity = zeros(nCompares, nTwindows, nCompares-1);
time_mdl1v4_ss13_predProbs = zeros(sum(y == 1), nTwindows, 2);

for iTime_window = 1:nTwindows
    
    tstart_idx = tWindow_starts(iTime_window);
    tend_idx = tstart_idx + tWindow_size;
    
    % compare 1 vs 4, 1 vs 1+3, and 1+3 vs 4
    for iCompare = 1:3
        
        if iCompare == 1
            class_labels = [0, 2]; % 1 v 4
            ignore_label = 1;
        elseif iCompare == 2
            class_labels = [0, 1]; % 1 v 1+3
            ignore_label = 2;
        else
            class_labels = [1, 2]; % 1+3 v 4
            ignore_label = 0;
        end
        
        
        % ------- Cross Validation Loop -------
        
        %cv = cvpartition(y_train_val, 'KFold', nFolds);
        cv = balanced_cv_split(y_train_val, 'KFolds', nFolds, 'TestSize', 0.1, 'RandomState', 23);
        cvMdls = {};
        for iFold = 1:cv.NumTestSets
            
            %train_idx = training(cv, iFold);
            %val_idx = test(cv, iFold);
            train_idx = cv.training{iFold};
            val_idx = cv.test{iFold};
            
            % PCA transform
            X_train = X_train_val(:,:,train_idx);
            y_train = y_train_val(train_idx);
            
            % compute trial averages and z-score
            X_train_avgs = arrayfun(@(i) squeeze(mean(X_train(:,:,y_train==i), 3)), class_labels, 'UniformOutput', false);
            X_train_avgs = cell2mat(X_train_avgs);
            X_train_avgs_mu = mean(X_train_avgs,2); 
            X_train_avgs_sd = std(X_train_avgs,[],2);
            X_train_avgs = (X_train_avgs - X_train_avgs_mu) ./ X_train_avgs_sd;
            components = pca(X_train_avgs');
            
            % project trial average components onto single trial data
            X_train_z = (X_train - X_train_avgs_mu) ./ X_train_avgs_sd;
            X_train_proj = components * reshape(X_train_z, size(X_train_z,1), size(X_train_z,2)*size(X_train_z,3));
            X_train_proj = reshape(X_train_proj, size(X_train_z,1), size(X_train_z,2), size(X_train_z,3));
            X_train = squeeze(mean(X_train_proj(:,tstart_idx:tend_idx,:),2))';
            
            % average over time period
            %X_train = squeeze(mean(X_train_val(:,tstart_idx:tend_idx,train_idx),2))';
            y_train = y_train_val(train_idx);
            
            cueDirs_train = cueDirs_train_val(train_idx);
            
            X_val = X_train_val(:,:,val_idx);
            X_val_z = (X_val - X_train_avgs_mu) ./ X_train_avgs_sd;
            X_val_proj = components * reshape(X_val_z, size(X_val_z,1), size(X_val_z,2)*size(X_val_z,3));
            X_val_proj = reshape(X_val_proj, size(X_val_z,1), size(X_val_z,2), size(X_val_z,3));
            X_val = squeeze(mean(X_val_proj(:, tstart_idx:tend_idx,:),2))';
            
            %X_val = squeeze(mean(X_train_val(:,tstart_idx:tend_idx,val_idx),2))';
            y_val = y_train_val(val_idx);
            
            cueDirs_val = cueDirs_train_val(val_idx);
            
            % concatenate cue direction as feature
            %X_train = [X_train, cueDirs_train];
            %X_train = cueDirs_train;
            
            %X_val = [X_val, cueDirs_val];
            %X_val = cueDirs_val;
            
            % ignore trials for class not being modeled in this loop
            % iteration
            X_train = X_train(y_train ~= ignore_label,:);
            y_train = y_train(y_train ~= ignore_label);
            
            X_val = X_val(y_val ~= ignore_label, :);
            y_val = y_val(y_val ~= ignore_label);
            
            % LDA
            mdl = fitcdiscr(X_train, y_train,'SaveMemory', 'on');
            
            cvMdls{iFold} = mdl;
            
            train_loss = resubLoss(mdl);
            time_train_loss(iCompare, iTime_window,iFold,:) = train_loss;
            train_preds = predict(mdl, X_train);
            
            
            val_loss = loss(mdl, X_val, y_val);
            time_val_loss(iCompare, iTime_window,iFold,:) = val_loss;
            val_preds = predict(mdl, X_val);
           
            
            % Specificity (i.e., accuracy per label)
            for iLabel = 1:length(class_labels)
                
                lab = class_labels(iLabel);
                
                val_tp = sum(val_preds == lab & y_val == lab);
                val_fn = sum(val_preds == lab & y_val ~= lab);
                
                time_val_specificity(iCompare, iTime_window, iFold, iLabel, :) = val_tp / (val_tp + val_fn);
                
                train_tp = sum(train_preds == lab & y_train == lab);
                train_fn = sum(train_preds == lab & y_train ~= lab);
                
                time_train_specificy(iCompare, iTime_window, iFold, iLabel, :) = train_tp / (train_tp + train_fn);
            end
            
        end
        
        % ------- Testing -------
        best_mdlIdx = time_val_loss(iCompare, iTime_window,:) == min(time_val_loss(iCompare, iTime_window,:));
        best_mdl = cvMdls{best_mdlIdx};
        
        % save best model weights
        train_mdl_weights(iCompare, iTime_window, :) = best_mdl.DeltaPredictor;
        
        X_test_t = squeeze(mean(X_test(:,tstart_idx:tend_idx,:), 2))';
        %X_test_t = [X_test_t, cueDirs_test];
        
        X_test_t = X_test_t(y_test ~= ignore_label, :);
        y_test_modeledTrials = y_test(y_test ~= ignore_label);
        
        test_loss = loss(best_mdl, X_test_t, y_test_modeledTrials);
        time_test_loss(iCompare, iTime_window) = test_loss;
        
        test_preds = predict(best_mdl, X_test_t);
        
        % Specificity (i.e., accuracy per label)
        for iLabel = 1:length(class_labels)
            
            lab = class_labels(iLabel);
            
            test_tp = sum(test_preds == lab & y_test_modeledTrials == lab);
            test_fn = sum(test_preds == lab & y_test_modeledTrials ~= lab);
            
            time_test_specificity(iCompare, iTime_window, iLabel, :) = test_tp / (test_tp + test_fn);
            
        end
        
        % ---- Test Model 1v4 model on 1+3 trials
        if iCompare == 1
            ss1_3_idx = y == ignore_label;
            ss1_3_y = y(ss1_3_idx);
            ss1_3_cueDirs = cueDirs(ss1_3_idx);
            ss1_3_X = squeeze(mean(X(:, tstart_idx:tend_idx, ss1_3_idx),2))';
            %ss1_3_X = [ss1_3_X, ss1_3_cueDirs];
            
            [~, ss1_3_X_predProbs] = predict(best_mdl, ss1_3_X);
            time_mdl1v4_ss13_predProbs(:, iTime_window, :) = ss1_3_X_predProbs;
        end
        
        
    end
    
end


classifier_results = [];
classifier_results.valLoss = time_val_loss;
classifier_results.valSpecificity = time_val_specificity;
classifier_results.testLoss = time_test_loss;
classifier_results.testSpecificity = time_test_specificity;
classifier_results.slidingWindow_startTimes = time(tWindow_starts);
classifier_results.windowSize_ms = tWindow_ms;
classifier_results.windowSize = tWindow_size;
classifier_results.compare_order = [0, 2; 0, 1; 1, 2];
classifier_results.train1v4_test13.predProbs = time_mdl1v4_ss13_predProbs;
classifier_results.train1v4_test13.col_labs = [0, 2];

save([saveDir, sprintf('sj%02d_voltage_classifyAcc.mat', sjNum)], 'classifier_results', '-v7.3');

end


%% Helper Functions

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