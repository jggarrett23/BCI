function spatial_SNR(iCon)

% addpath(genpath('/home/garrett/WTF_Bike/Analysis_Scripts/'))
% 
% 
% cd('/home/garrett/eeglab14_1_2b')
% eeglab
% close all


cd('/home/garrett/WTF_Bike/Analysis_Scripts/')

subjects = [1:8,10:35];
conds = 1:2;

root = '/home/garrett/WTF_Bike/';
hilbDir = [root 'HILBERT/'];
dataDir = [root 'Data_Compiled/'];
outDir = [root 'Overall_Images/'];
head_file = [root 'Analysis_Scripts/WTF_spline.spl'];

x = linspace(-500,2000,625);
baseline = (dsearchn(x',-200):dsearchn(x',0));

window_length = 1; step_size = 1;%window_length*.5; %sliding windows of 13 samples (~50ms) with %50 overlap
%nWindows = 0:step_size:(length(x)-window_length);
nWindows = 1; %do if just want to look at specific time period

load([hilbDir 'sj01_exerCon01_changeDect_EEG.mat'],'eeg');

elec_pos = eeg.chanInfo;

clear eeg

if iCon == 1
    cond_name = 'Rest';
else
    cond_name = 'Exercise';
end

%load electrode SNR for all subs
load([dataDir sprintf('exerCon%02d_SNR_all.mat',iCon)],'electrode_snr')
for iView = 1:2
    
    if iView == 1
        view = 'back';
    else
        view = 'front';
    end
    
    i = 1;
    for iWindow = nWindows
        %woi = floor([iWindow + 1 : window_length + iWindow]);
        
        %just delay period activity
        woi = dsearchn(x',500):dsearchn(x',2000);
        
        %baseline correct 0-500ms, then normalize between [0 1]
        norm_bcElecSnr = [];
        for iSub = 1:size(electrode_snr,1)
            bc_electrodeSnr = electrode_snr(iSub,:,:) - mean(electrode_snr(iSub,:,baseline),3);
            
            
            %average over timepoints in window of interest
            
            avg_bc_ElSnr = squeeze(mean(bc_electrodeSnr(:,:,woi),3));
            
            norm_bcElecSnr(iSub,:) = (avg_bc_ElSnr - min(avg_bc_ElSnr))/(max(avg_bc_ElSnr)-min(avg_bc_ElSnr));
        end
        
        %average over subjects
        avg_electrodeSNR = squeeze(mean(norm_bcElecSnr,1));
        
        
        
        h=figure(iCon);
        %drawnow
        headplot(avg_electrodeSNR,head_file,'meshfile','/home/garrett/eeglab14_1_2b/functions/resources/mheadnew.mat',...
            'view',view,'electrodes','off','maplimits', [0.4 0.7],...
            'cbar',0);
        
        %just delay period
        title(sprintf('%s Condition Delay Period SNR', cond_name))
        
        %title(sprintf('%s Condition Time: %3.0f ms',cond_name,x(woi)))
        
%         if i == 1
%             gif([outDir sprintf('exerCond%02d_%s_view_spatialSNR.gif',iCon,view)],'frame',gcf,...
%                 'DelayTime',0.15)
%         else
%             gif
%         end
        
        i = i+1;
        
        %saveas(h,[outDir sprintf('exerCond%02d_%sView_SNR.jpg',iCon,view)]);
        saveas(h,['/home/garrett/WTF_Bike/tmpDir/' sprintf('newSNR_exerCond%02d_%sView.jpg',iCon,view)])
        close all
    end
end



return