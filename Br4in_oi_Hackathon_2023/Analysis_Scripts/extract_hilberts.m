function extract_hilberts (sjNum, condition, freq_band)

%% Settings
parent_folder = '/work/garrett/DT_CDA/';
hilbDir = [parent_folder 'Hilbert/EEG/'];
spectraDir = [parent_folder 'Hilbert/Spectra/'];

% load EEG
load([hilbDir sprintf('sj%02d_cond%02d_HILB.mat',sjNum,condition)],'EEG');

%% Baseline correct
time = EEG.times;
baseline_idx = dsearchn(time',-500):dsearchn(time',-200);
data=double(EEG.data);
%data = data-mean(data(:,baseline_idx,:),3);

%% Butterworth Filter
filter_order = 3;

type = 'bandpass';
[z1,p1] = butter(filter_order,freq_band./(EEG.srate/2),type);
tempEEG = NaN(size(data,1),EEG.pnts,size(data,3));
for iChan = 1:size(data,1)
   for iTrial = 1:size(data,3)
       dataFilt1 = filtfilt(z1,p1,data(iChan,:,iTrial));
       tempEEG(iChan,:,iTrial) = dataFilt1;
   end
end

%% Apply Hilberts
eegs = [];
for iChan = 1:size(tempEEG,1)
   for iTrial = 1:size(tempEEG,3)
      eegs(iTrial,iChan,:) = hilbert(squeeze(tempEEG(iChan,:,iTrial)));
   end
end


%% Save spectra data
spectra.complex = eegs;
%spectra.complex_bc = eegs_bc;
spectra.time = time;
spectra.freq_band = freq_band;
spectra.chanlocs = EEG.chanlocs;
spectra.trialInfo = EEG.newTrialInfo;

if min(freq_band) == 8 && max(freq_band) == 12
    freq_name = 'ALPHA';
elseif min(freq_band) == 4 && max(freq_band) == 7
    freq_name = 'THETA';
elseif min(freq_band) == 13 && max(freq_band) == 20
    freq_name = 'LBETA';
elseif min(freq_band) == 21 && max(freq_band) == 30
    freq_name = 'HBETA';
elseif min(freq_band) == 31 && max(freq_band) == 38
    freq_name = 'LGAMMA';
elseif min(freq_band) == 1 && max(freq_band) == 3
    freq_name = 'DELTA';
end
    

save([spectraDir sprintf('sj%02d_con%02d_%s_spectra.mat',sjNum,condition,freq_name)],'spectra','-v7.3');

end