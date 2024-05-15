function Hilb_classify_trialType (sjNum)

parentDir = '/work/garrett/CDA_Bike/';
dataDir = [parentDir 'Hilbert/Spectra/'];
saveDir = [parentDir, 'Classifier_Results/'];


%% Load in Spectral Data
bands_complex = [];
all_trialTypes = {};
allData = {};
cueDirs = [];
for iCon = 1:2
    
    con_bands_complex = [];
    for iBand = 1:6
        
        if iBand == 1
            band_name = 'THETA';
        elseif iBand == 2
            band_name = 'ALPHA';
        elseif iBand == 3
            band_name = 'LBETA';
        elseif iBand == 4
            band_name = 'HBETA';
        elseif iBand == 5
            band_name = 'LGAMMA';
        elseif iBand == 6
            band_name = 'DELTA';
        end
        
        load([dataDir sprintf('sj%02d_con%02d_%s_spectra.mat', sjNum, iCon, band_name)], 'spectra');
        
        con_bands_complex(iBand, :, :, :) = spectra.complex;
        
    end
    
    trialInfo = spectra.trialInfo;
    
    % extract specific trials
    ss1_trials = [trialInfo.SetSize] == 1;
    ss4_trials = intersect(find([trialInfo.SetSize] == 4), find([trialInfo.Distractor_Present] ~= 1));
    ss13_trials = intersect(find([trialInfo.SetSize] == 4), find([trialInfo.Distractor_Present]));
    
    y = zeros(length(ss1_trials),1); % ss1
    y(ss13_trials) = 1; % ss 1+3
    y(ss4_trials) = 2; % ss 4
    
    all_trialTypes{iCon} = y;
    
    allData{iCon} = con_bands_complex;
    
end

rest_complex = double(allData{1});
ex_complex = double(allData{2});

rest_trialLabels = all_trialTypes{1};
ex_trialLabels = all_trialTypes{2};

y = [rest_trialLabels; ex_trialLabels];

time = spectra.time;
baseline_idx = dsearchn(time',-200):dsearchn(time',0);

rest_pow = abs(rest_complex).^2;
ex_pow = abs(ex_complex).^2;

% normalize electrodes activity over time
%bands_pow = (bands_pow - mean(bands_pow,[2,3]))./std(bands_pow,0,[2,3]);

rest_pow = rest_pow - mean(rest_pow(:, :,:,baseline_idx), 4);
ex_pow = ex_pow - mean(ex_pow(:, :,:,baseline_idx), 4);


nFreqs = size(rest_pow,1);
%% Classification

% combine frequencies 
%X = reshape(X, [1, size(X,1)*size(X,2), size(X,3), size(X,4)]);

trial_types = {'1', '1+3', '4'};

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

rng(3);

nChans = size(rest_pow,3);

nFolds = 10;
nCompares = 3;

cv_time_train_loss = zeros(nFreqs, nCompares, nFolds, length(time));

rest_cv_time_val_loss = cv_time_train_loss;
ex_cv_time_val_loss = cv_time_train_loss;

rest_cv_time_train1v4_test1_3_time_probs = zeros(nFreqs, nFolds, sum(rest_trialLabels==1), length(time), 2);
ex_cv_time_train1v4_test1_3_time_probs = zeros(nFreqs, nFolds, sum(ex_trialLabels==1), length(time), 2);


for iBand = 1:nFreqs
    
    rest_data = squeeze(rest_pow(iBand,:,:,:));
    ex_data = squeeze(ex_pow(iBand,:,:,:));
    
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
            min_trials = min([length(rest_train_idx), length(ex_train_idx)]);
            rest_train_idx = rest_train_idx(1:min_trials);
            ex_train_idx = ex_train_idx(1:min_trials);
            
            rest_Xtrain = rest_currentCompare_data(rest_train_idx, :, :);
            rest_y_train = rest_currentComp_labels(rest_train_idx);
            
            ex_Xtrain = ex_currentCompare_data(ex_train_idx,:,:);
            ex_y_train = ex_currentComp_labels(ex_train_idx);
            
            % train on both conditions
            X_train = cat(1, rest_Xtrain, ex_Xtrain);
            y_train = [rest_y_train; ex_y_train];
            
            rest_Xval = rest_currentCompare_data(rest_val_idx,:,:);
            rest_y_val = rest_currentComp_labels(rest_val_idx);
            
            ex_Xval = ex_currentCompare_data(ex_val_idx,:,:);
            ex_y_val = ex_currentComp_labels(ex_val_idx);
            
            % Loop over time
            for iTime = 1:length(time)
                
                % LDA
                mdl = fitcdiscr(squeeze(X_train(:,:,iTime)), y_train,'SaveMemory', 'on');
                
                cv_time_train_loss(iBand, iCompare, iFold, iTime) = resubLoss(mdl);
                
                rest_val_loss = loss(mdl, squeeze(rest_Xval(:,:,iTime)), rest_y_val);
                ex_val_loss = loss(mdl, squeeze(ex_Xval(:,:,iTime)), ex_y_val);
                
                rest_cv_time_val_loss(iBand, iCompare, iFold, iTime) = rest_val_loss;
                ex_cv_time_val_loss(iBand, iCompare, iFold, iTime) = ex_val_loss;
                
                
                % ---- Test Model 1v4 model on 1+3 trials for rest and exercise
                if iCompare == 1
                    rest_ss1_3_idx = rest_trialLabels == ignore_label;
                    rest_ss1_3_y = rest_trialLabels(rest_ss1_3_idx);
                    
                    ex_ss1_3_idx = ex_trialLabels == ignore_label;
                    ex_ss1_3_y = ex_trialLabels(ex_ss1_3_idx);
                    
                    rest_ss1_3_X = rest_data(rest_ss1_3_idx,:,:);
                    
                    
                    ex_ss1_3_X = ex_data(ex_ss1_3_idx,:,:);
                    
                    
                    [~, rest_ss1_3_X_predProbs] = predict(mdl, squeeze(rest_ss1_3_X(:,:,iTime)));
                    rest_cv_time_train1v4_test1_3_time_probs(iBand, iFold, :, iTime, :) = rest_ss1_3_X_predProbs;
                    
                    [~, ex_ss1_3_X_predProbs] = predict(mdl, squeeze(ex_ss1_3_X(:,:,iTime)));
                    ex_cv_time_train1v4_test1_3_time_probs(iBand, iFold, :, iTime, :) = ex_ss1_3_X_predProbs;
                end
                
            end
        end
    end
end

classifier_results = [];
classifier_results.trainLoss = cv_time_train_loss;
classifier_results.valLoss.rest = rest_cv_time_val_loss;
classifier_results.valLoss.ex = ex_cv_time_val_loss;
classifier_results.compare_order = [0, 2; 0, 1; 1, 2];
classifier_results.train1v4_test13.predProbs.rest = rest_cv_time_train1v4_test1_3_time_probs;
classifier_results.train1v4_test13.predProbs.ex = ex_cv_time_train1v4_test1_3_time_probs;
classifier_results.train1v4_test13.col_labs = [0, 2];
classifier_results.time = time;
classifier_results.nFreqs = nFreqs;
classifier_results.freq_bands = [4,7; 8,12; 13, 20; 21, 30; 31, 38; 1, 3];

save([saveDir, sprintf('sj%02d_trainBoth_hilb_classifyAcc.mat', sjNum)], 'classifier_results', '-v7.3');
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