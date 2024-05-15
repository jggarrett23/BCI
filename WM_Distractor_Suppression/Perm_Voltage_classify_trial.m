function Perm_Voltage_classify_trial(sjNum)

%% Set Up
parentDir = '/work/garrett/CDA_Bike/';
dataDir = [parentDir 'ICA_Preprocess/Epoch/'];
saveDir = [parentDir 'Classifier_Results/'];

allData = {};
all_trialTypes = {};
for iCon = 1:2
    
    % load data
    load([dataDir sprintf('sj%02d_cond%02d_ICA_EEG.mat', sjNum, iCon)], 'EEG');
    
    % electrodes x time points x trials
    data = EEG.data;
    
    allData{iCon} = data;
    
    % extract specific trials
    ss1_trials = [EEG.newTrialInfo.SetSize] == 1;
    ss4_trials = intersect(find([EEG.newTrialInfo.SetSize] == 4), find([EEG.newTrialInfo.Distractor_Present] ~= 1));
    ss13_trials = intersect(find([EEG.newTrialInfo.SetSize] == 4), find([EEG.newTrialInfo.Distractor_Present]));
    
    y = zeros(length(ss1_trials),1); % ss1
    y(ss13_trials) = 1; % ss 1+3
    y(ss4_trials) = 2; % ss 4
    
    all_trialTypes{iCon} = y;
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

rest_data = double(allData{1});
ex_data = double(allData{2});

rest_data = permute(rest_data(chans_oi_idx,:,:), [3, 1, 2]);
ex_data = permute(ex_data(chans_oi_idx,:,:), [3, 1, 2]);

rest_trialLabels = all_trialTypes{1};
ex_trialLabels = all_trialTypes{2};

baseline = [-200, 0];

% baseline correct data
baseline_idx = dsearchn(time',baseline(1)):dsearchn(time',baseline(2));

rest_data = rest_data - mean(rest_data(:, :, baseline_idx),3);
ex_data = ex_data - mean(ex_data(:, :, baseline_idx),3);


nChans = size(rest_data,2);

nFolds = 250;
nCompares = 3;
nBlockIters = 100;
nTrials_perMiniBlock = 20;

cv_time_train_loss = zeros(nCompares, nFolds, length(time));

rest_cv_time_val_loss = cv_time_train_loss;
ex_cv_time_val_loss = cv_time_train_loss;

rest_cv_time_train1v4_test1_3_time_probs = zeros(nFolds, sum(rest_trialLabels==1), length(time), 2);
ex_cv_time_train1v4_test1_3_time_probs = zeros(nFolds, sum(ex_trialLabels==1), length(time), 2);

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
    
    % Extract data for current comparison
    rest_currentCompare_data = rest_data(rest_trialLabels ~= ignore_label, :, :);
    ex_currentCompare_data = ex_data(ex_trialLabels ~= ignore_label, :,:);
    
    rest_currentComp_labels = rest_trialLabels(rest_trialLabels ~= ignore_label);
    ex_currentComp_labels = ex_trialLabels(ex_trialLabels ~= ignore_label);
    
    % ------- Cross Validation Loop -------
   
    rest_cv = balanced_cv_split(rest_currentComp_labels, 'KFolds', nFolds, 'TestSize', 0.1, 'RandomState', 23);
    ex_cv = balanced_cv_split(ex_currentComp_labels, 'KFolds', nFolds, 'TestSize', 0.1, 'RandomState', 23);
    
    cvMdls = {};
    for iFold = 1:rest_cv.NumTestSets
        
        % Split rest and exercise data equally
        rest_train_idx = rest_cv.training{iFold};
        rest_val_idx = rest_cv.test{iFold};
        
        ex_train_idx = ex_cv.training{iFold};
        ex_val_idx = ex_cv.test{iFold};
        
        % Retain minimum number of trials for each condition
        min_train_trials = min([length(rest_train_idx), length(ex_train_idx)]);
        rest_train_idx = rest_train_idx(1:min_train_trials);
        ex_train_idx = ex_train_idx(1:min_train_trials);
        
        rest_Xtrain = rest_currentCompare_data(rest_train_idx, :, :);
        rest_y_train = rest_currentComp_labels(rest_train_idx);
        
        ex_Xtrain = ex_currentCompare_data(ex_train_idx,:,:);
        ex_y_train = ex_currentComp_labels(ex_train_idx);
        
        
        % ------- Mini-Block Averaging Loop -------
        
        rest_blockX_train = zeros(nBlockIters, 2, nChans, length(time));
        ex_blockX_train = zeros(nBlockIters, 2, nChans, length(time));
        rest_blockY_train = zeros(nBlockIters, 2, 1);
        ex_blockY_train = zeros(nBlockIters, 2, 1);
        
        for iIter = 1:nBlockIters
            
            rest_train_shuffIdx = randperm(min_train_trials);
            ex_train_shuffIdx = randperm(min_train_trials);
            
            rest_train_shuff_y = rest_y_train(rest_train_shuffIdx);
            ex_train_shuff_y = ex_y_train(ex_train_shuffIdx);
            
            rest_train_shuff_X = rest_Xtrain(rest_train_shuffIdx,:,:);
            ex_train_shuff_X = ex_Xtrain(ex_train_shuffIdx,:,:);
            
            % pull out first nTrials for averaging
            for i=1:2
                rest_class_idx = find(rest_train_shuff_y==class_labels(i));
                ex_class_idx = find(ex_train_shuff_y == class_labels(i));
                
                rest_blockX_train(iIter, i, :, :) = mean(rest_train_shuff_X(rest_class_idx(1:floor(nTrials_perMiniBlock/2)), :, :));
                ex_blockX_train(iIter, i, :, :) = mean(ex_train_shuff_X(ex_class_idx(1:floor(nTrials_perMiniBlock/2)), :, :));
                
                
                rest_blockY_train(iIter, i) = class_labels(i);
                ex_blockY_train(iIter, i) = class_labels(i);
            end
        end
        
        rest_Xtrain = reshape(rest_blockX_train, nBlockIters*2, nChans, length(time));
        ex_Xtrain = reshape(ex_blockX_train, nBlockIters*2, nChans, length(time));
        
        rest_y_train = reshape(rest_blockY_train, nBlockIters*2, 1);
        ex_y_train = reshape(ex_blockY_train, nBlockIters*2, 1);
        
        
        % train on both conditions
        X_train = cat(1, rest_Xtrain, ex_Xtrain);
        y_train = [rest_y_train; ex_y_train];
        
        rest_Xval = rest_currentCompare_data(rest_val_idx,:,:);
        rest_y_val = rest_currentComp_labels(rest_val_idx);
        
        ex_Xval = ex_currentCompare_data(ex_val_idx,:,:);
        ex_y_val = ex_currentComp_labels(ex_val_idx);
        
        % SHUFFLE TRAINING LABELS
        shuff_idx = randperm(length(y_train));
        y_train = y_train(shuff_idx);
        
        % Loop over time
        for iTime = 1:length(time)
            
            % LDA
            mdl = fitcdiscr(squeeze(X_train(:,:,iTime)), y_train,'SaveMemory', 'on');
            
            cv_time_train_loss(iCompare, iFold, iTime) = resubLoss(mdl);
            
            rest_val_loss = loss(mdl, squeeze(rest_Xval(:,:,iTime)), rest_y_val);
            ex_val_loss = loss(mdl, squeeze(ex_Xval(:,:,iTime)), ex_y_val);
            
            rest_cv_time_val_loss(iCompare, iFold, iTime) = rest_val_loss;
            ex_cv_time_val_loss(iCompare, iFold, iTime) = ex_val_loss;
            
            
            % ---- Test Model 1v4 model on 1+3 trials for rest and exercise
            if iCompare == 1
                rest_ss1_3_idx = rest_trialLabels == ignore_label;
                rest_ss1_3_y = rest_trialLabels(rest_ss1_3_idx);
                
                ex_ss1_3_idx = ex_trialLabels == ignore_label;
                ex_ss1_3_y = ex_trialLabels(ex_ss1_3_idx);
                
                rest_ss1_3_X = rest_data(rest_ss1_3_idx,:,:);
                
                
                ex_ss1_3_X = ex_data(ex_ss1_3_idx,:,:);
                
                
                [~, rest_ss1_3_X_predProbs] = predict(mdl, squeeze(rest_ss1_3_X(:,:,iTime)));
                rest_cv_time_train1v4_test1_3_time_probs(iFold, :, iTime, :) = rest_ss1_3_X_predProbs;
                
                [~, ex_ss1_3_X_predProbs] = predict(mdl, squeeze(ex_ss1_3_X(:,:,iTime)));
                ex_cv_time_train1v4_test1_3_time_probs(iFold, :, iTime, :) = ex_ss1_3_X_predProbs;
            end
            
        end
    end
end

perm_classifier_results = [];
perm_classifier_results.trainLoss = cv_time_train_loss;
perm_classifier_results.valLoss.rest = rest_cv_time_val_loss;
perm_classifier_results.valLoss.ex = ex_cv_time_val_loss;
perm_classifier_results.compare_order = [0, 2; 0, 1; 1, 2];
perm_classifier_results.train1v4_test13.predProbs.rest = rest_cv_time_train1v4_test1_3_time_probs;
perm_classifier_results.train1v4_test13.predProbs.ex = ex_cv_time_train1v4_test1_3_time_probs;
perm_classifier_results.train1v4_test13.col_labs = [0, 2];
perm_classifier_results.time = time;

save([saveDir, sprintf('sj%02d_perm_trainBoth_voltage_classifyAcc.mat', sjNum)], 'perm_classifier_results', '-v7.3');



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

uniq_labels = unique(labels);
nClasses = length(uniq_labels);

if nClasses == 2 && (max(labels) - min(labels)) ~= 1
   labels(labels==max(labels)) = min(labels) + 1; 
   uniq_labels = unique(labels);
end


[counts, ~] = histcounts(labels, min(labels):max(labels)+1);


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