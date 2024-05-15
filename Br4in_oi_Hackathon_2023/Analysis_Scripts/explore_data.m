function explore_data

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
   nPnts = size(eeg_data,2);
   
   %% Epoch data
   
   %{

    Trial structure: 
    Each run began with a 10s delay. Next, each trial started with a 3s
    pause. Subjects were asked to fixate for 7s on flicker.

    % used a 200ms sliding window to classify stimulation frequencies

    %}
   
   % use on/off triggers to find contiguous seconds of stimulus flashing
   [epoch_idxs, y] = parse_data(triggers);
   
   epoch_idxs = [1, epoch_idxs, nPnts];
   stimOff_data = {}; crop_cnt = 1;
   for iDelay_start = 1:2:length(epoch_idxs)
       crop_start = epoch_idxs(iDelay_start);
       crop_end = epoch_idxs(iDelay_start+1)-1;
       stimOff_data{crop_cnt} = eeg_data(:,crop_start:crop_end);
       crop_cnt = crop_cnt+1;
   end
   
   % truncate off trials to minimum length
   nFlick_offTrials = crop_cnt - 1;
   min_offLength = min(cellfun(@(x) size(x,2), stimOff_data));
   stimOff_data{1} = stimOff_data{1}(:,end-min_offLength+1:end);
   stimOff_data{end} = stimOff_data{end}(:,1:min_offLength);
   stimOff_data = cell2mat(stimOff_data);
   stimOff_data = reshape(stimOff_data, nChans, min_offLength, nFlick_offTrials);
   
   
   %% FFT Check
   
   srate = 256; % sampling rate
   L = trial_length; % signal length
   NFFT = 2^nextpow2(L); % number of points for fft
   stim_onFreqs = srate/2*linspace(0,1,NFFT/2+1); % frequencies that can be resolved
   
   stimOn_spectra = zeros(nChans, NFFT, nFlick_onTrials);
   for iTrial = 1:nFlick_onTrials
       stimOn_spectra(:,:,iTrial) = fft(squeeze(stimOn_data(:,:,iTrial)), NFFT, 2)/L; % divide by L to project back into orignal data units
   end
   stimOn_spectra = stimOn_spectra(:,1:length(stim_onFreqs),:); % truncate spectra
   
   NFFT = 2^nextpow2(min_offLength);
   stim_offFreqs = srate/2*linspace(0,1,NFFT/2+1);
   
   stim_offSpectra = zeros(nChans, NFFT, nFlick_offTrials);
   for iTrial = 1:size(stim_offSpectra,3)
       stim_offSpectra(:,:,iTrial) = fft(squeeze(stimOff_data(:,:,iTrial)), NFFT, 2)/min_offLength;
   end
   
   stim_offSpectra = stim_offSpectra(:,1:length(stim_offFreqs),:);
   
   % average over trials for each flicker
   figure()
   
   for iFlick = 1:4
       
       subplot(2,2,iFlick)
       
       evoked_pow = abs(mean(stimOn_spectra(:,:,trial_labels==iFlick), 3)).^2;
       
       % PSD Visualization
       plot(stim_onFreqs, evoked_pow);
       xlim([1, 30]);
       ylabel('Pow (\muV^2)');
       xlabel('Frequency (Hz)');
       box off
       title(sprintf('%d Hz', flick_freqs(iFlick)));
       sgtitle(sprintf('Subject %d', iSub));
   end
   
   %% First pass classifier
   
   % using SNR
   stimOn_flick_idxs = arrayfun(@(i) dsearchn(stim_onFreqs',i), flick_freqs);
   stimOff_flick_idxs = arrayfun(@(i) dsearchn(stim_offFreqs',i), flick_freqs);
   
   stimOn_total_pow = abs(stimOn_spectra).^2;
   stimOff_total_pow = abs(stim_offSpectra).^2;
   
   % Stim On Trials
   
   % signal
   flick_pow = arrayfun(@(i) squeeze(mean(stimOn_total_pow(:,[i-1, i, i+1],:),2)), stimOn_flick_idxs, 'UniformOutput', false);
   flick_pow = cell2mat(flick_pow);
   flick_pow = reshape(flick_pow, nFlicks, nChans, nFlick_onTrials);
   
   % noise
   flick_noise = arrayfun(@(i) squeeze(std(stimOn_total_pow(:,[i-4,i-3,i+3,i-4],:),[],2)), stimOn_flick_idxs, 'UniformOutput', false);
   flick_noise = cell2mat(flick_noise);
   flick_noise = reshape(flick_noise, nFlicks, nChans, nFlick_onTrials);
   
   stimOn_flick_snr = flick_pow ./ flick_noise;
   
   % Stim Off Trials
   
   % signal
   flick_pow = arrayfun(@(i) squeeze(mean(stimOff_total_pow(:,[i-1, i, i+1],:),2)), stimOff_flick_idxs, 'UniformOutput', false);
   flick_pow = cell2mat(flick_pow);
   flick_pow = reshape(flick_pow, nFlicks, nChans, nFlick_offTrials);
   
   % noise
   flick_noise = arrayfun(@(i) squeeze(std(stimOff_total_pow(:,[i-4,i-3,i+3,i-4],:),[],2)), stimOff_flick_idxs, 'UniformOutput', false);
   flick_noise = cell2mat(flick_noise);
   flick_noise = reshape(flick_noise, nFlicks, nChans, nFlick_offTrials);
   
   stimOff_flick_snr = flick_pow ./ flick_noise;
   
   X = cat(3, stimOn_flick_snr, stimOff_flick_snr);
   trial_labels = [zeros(nFlick_offTrials, 1); trial_labels];
   nTrials = length(trial_labels);
   X = reshape(X, nFlicks*nChans, nTrials);
   
   rng(23)
   
   shuffIdx = randperm(nTrials);
   X = X(:,shuffIdx);
   trial_labels = trial_labels(shuffIdx);
   
   % LOO CV
   cv_results = zeros(nTrials,1);
   cv_confusions = [];
   for iTrial = 1:nTrials
       X_test = X(:,iTrial)';
       y_test = trial_labels(iTrial);
       X_train = X(:,setdiff(1:nTrials,iTrial))';
       y_train = trial_labels(setdiff(1:nTrials,iTrial));
       
       mdl = fitcdiscr(X_train, y_train, 'SaveMemory', 'on');
       train_loss = resubLoss(mdl);
       test_loss = loss(mdl, X_test, y_test);
       cv_results(iTrial,:) = test_loss;
       
       train_preds = predict(mdl,X_train);
       
       cv_confusions(iTrial,:,:) = confusionmat(y_train, train_preds);
   end
   
   fprintf('Sj %d: LOO-CV LDA Accuracy: %0.2f%%\n', iSub, mean(cv_results)*100);
   
   figure;
   imagesc(squeeze(mean(cv_confusions)));
   
end

foo = 0;

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