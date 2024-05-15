function extract_fft (sjNum, con)

cd /home/garrett/eeglab14_1_2b
eeglab
close all
cd /work/garrett/DT_CDA/Analysis_Scripts


parentDir = '/work/garrett/DT_CDA/';
dataDir = [parentDir, 'EEG/'];
saveDir = [parentDir 'FFT/'];

% load data
load([dataDir sprintf('sj%02d_cond%02d_EEG.mat', sjNum, con)], 'EEG');

data = double(EEG.data);
data = permute(data, [3,1,2]);

chanInfo = EEG.chanlocs;

time = EEG.times;

po_o_elecs = {'PO7','PO8','PO3','PO4','POz','O1','O2','Oz',...
    'P7','P8','P5','P6','P3','P4','P1','P2','Pz'};

po_o_elecsIdx = cellfun(@(x) find(strcmp(x,{chanInfo.labels})), po_o_elecs);

% define time window parameters
tWindow_ms = 350; % size of time window in ms
wOverlap = 0.95; % amount of overlap between time windows

tWindow_size = length(dsearchn(time', 0):dsearchn(time', tWindow_ms));

tOverlap_step = floor(tWindow_size * (1-wOverlap));

if ~tOverlap_step
    tOverlap_step = 1;
end

tWindow_starts = 1:tOverlap_step:length(time)-tWindow_size;
nTwindows = length(tWindow_starts);


Fs = EEG.srate;
L = tWindow_size;
NFFT = 2^nextpow2(L);
freqs = Fs/2*linspace(0,1,NFFT/2+1);


% do fft on one slice of data to preallocation arrays
s = squeeze(fft(squeeze(data(1,:,1:tWindow_size)), NFFT, 2)/L);

spectra = zeros(size(data,1), nTwindows, size(s,1), size(s,2));
for iEpoch = 1:size(data,1)
    
    for iTwindow = 1:nTwindows
        
        tstart_idx = tWindow_starts(iTwindow);
        tend_idx = tstart_idx + tWindow_size - 1;
        
        trial_data = squeeze(data(iEpoch,:,tstart_idx:tend_idx));
        spectra(iEpoch,iTwindow,:,:) = fft(trial_data, NFFT, 2)/L;
        
    end
end

spectra_tr=spectra(:,:,:,1:length(freqs));

% extract specific trials
ss1_trials = [EEG.newTrialInfo.SetSize] == 1;
ss4_trials = intersect(find([EEG.newTrialInfo.SetSize] == 4), find([EEG.newTrialInfo.Distractor_Present] ~= 1));
ss13_trials = intersect(find([EEG.newTrialInfo.SetSize] == 4), find([EEG.newTrialInfo.Distractor_Present]));


fft_results = [];
fft_results.spectra = spectra_tr;
fft_results.freqs = freqs;
fft_results.time = time;
fft_results.twindow_starts = tWindow_starts;
fft_results.tWindow_size = tWindow_size;
fft_results.tWindow_ms = tWindow_ms;
fft_results.wOverlap = wOverlap;
fft_results.chanlocs = chanInfo;
fft_results.labels.ss1 = ss1_trials;
fft_results.labels.ss4  = ss4_trials;
fft_results.labels.ss13  = ss13_trials;



save([saveDir, sprintf('sj%02d_cond%02d_fft.mat', sjNum, con)], 'fft_results', '-v7.3');


end


