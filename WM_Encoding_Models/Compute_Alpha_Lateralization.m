%{
Compute_Alpha_Lateralization
Author: Tom Bullock
Date: 05.27.20

%}
function Compute_Alpha_Lateralization
% load EEGLAB (if needed)
% cd('/home/garrett/eeglab14_1_2b')
% eeglab
% close all
cd /home/garrett/WTF_Bike/Analysis_Scripts

%set dirs
parentDir = '/home/garrett/WTF_Bike/';
sourceDir = [parentDir 'Data_Compiled/'];
hilbDir = [parentDir 'HILBERT/'];
plotDir = [parentDir 'Overall_Images/'];

% load data
%load([sourceDir '/' 'Bandpassed_Data_ALpha.mat'])
load([hilbDir 'sep_eegs_all_ALPHA.mat'],'sep_eegs');

% conditions x subs x locations x electrodes x timepoints (625)
hilbData = sep_eegs;
% something weird going on with sj 8 rest condition data, getting NaNs
%hilbData = hilbData([1:7,9:end],:,:,:,:);

% create times vector
times = linspace(-1000,2500,size(hilbData,5));

epoch_oi = (dsearchn(times',-500):dsearchn(times',2000));

hilbData = sep_eegs(:,:,:,:,epoch_oi);

times = times(epoch_oi);

%use PO7, P7, P5 and PO8, P6, P8 (Thut)
chanlocs = load('/home/garrett/WTF_Bike/Analysis_Scripts/64_chan_loc.mat');
chanlocs = chanlocs.elec_pos;
right_chans = {'PO8','P6','P8'};
left_chans = {'PO7','P7','P5'};

leftHemiElects = zeros(1,length(left_chans)); left_cnt = 1;
rightHemiElects = zeros(1,length(right_chans)); right_cnt = 1;
for iChan = 1:length(chanlocs)
    if ismember(chanlocs(iChan).labels,left_chans)
        leftHemiElects(left_cnt) = iChan;
        left_cnt = left_cnt + 1;
        
    elseif ismember(chanlocs(iChan).labels,right_chans)
        rightHemiElects (right_cnt) = iChan;
        right_cnt = right_cnt + 1;
        
    end
    
end

leftStimLocs = 3:6;
rightStimLocs = [1,2,7,8];


% % set posterior channels for normalization
% poElects = [25 26 30 62 63];
% pElects = [22 21 20 31 57 58 59];
% oElects = [27 29 64];
% theseChans = [poElects,oElects,pElects];

% set time segments for analysis
% set time segments for analysis
thisTime=-200; % start baseline period
[~, baseStart] = min(abs(thisTime-times));
thisTime=0; % probe onset
[~, probeOn] = min(abs(thisTime-times));
thisTime=200; % probe
[~, probeFinal50ms] = min(abs(thisTime-times));
thisTime=250; % probe offset
[~, probeOff] = min(abs(thisTime-times));
thisTime=500;
[~, retPlotStart] = min(abs(thisTime-times));
thisTime=2000; % end retention period
[~, retEnd] = min(abs(thisTime-times));

% baseline correct to mean of whole baseline period
allBand = hilbData - mean(hilbData(:,:,:,:,baseStart:probeOn-1),5);

% convert bandpassed data from complex numbers to to power
allBand = abs(allBand).^2;

% loop through and plot differnet time segments
latIdx = [];
for timeSegmentToPlot=1:2
    
    % time segment settings
    if timeSegmentToPlot==1
        thisTimeSegment = probeOn:probeOff;
        thisTimeSegmentLabel = 'Stimulus On (0.2-0.25s)';
        thisTimeSegmentTitle = 'Probe';
    elseif timeSegmentToPlot==2
        thisTimeSegment = retPlotStart:retEnd;
        thisTimeSegmentLabel = 'Renetion Period (0.5-2s)';
        thisTimeSegmentTitle = 'Early Retention';
    end
    
    %     % condition loop
    %     h=figure('units','normalized','outerposition',[0.1 0.1 .9 1]);
    
    for iCond=1:2
        
        allIpsi = (mean(mean(mean(allBand(:,iCond,leftStimLocs,leftHemiElects,thisTimeSegment),3),4),5) + mean(mean(mean(allBand(:,iCond,rightStimLocs,rightHemiElects,thisTimeSegment),3),4),5))./2;
        
        allContra = (mean(mean(mean(allBand(:,iCond,leftStimLocs,rightHemiElects,thisTimeSegment),3),4),5) + mean(mean(mean(allBand(:,iCond,rightStimLocs,leftHemiElects,thisTimeSegment),3),4),5))./2;
        
        latIdx(:,iCond,timeSegmentToPlot) = (allIpsi - allContra)./ (allIpsi +allContra);
        
    end
    
end

% generate bar plots for lateralization at each time segment
h=figure('units','normalized','outerposition',[0.1 0.1 .5 1.2]);
for iPlot=1:2
    
    if      iPlot==1; thisTimeSegmentLabel = 'Stimulus (0-0.25s)';
    elseif  iPlot==2; thisTimeSegmentLabel = 'Retention (0.5-2s)';
    end
    
    subplot(1,2,iPlot)
    
    for i=1:2
        if       i==1; thisColor = 'g';
        elseif   i==2; thisColor = 'b';
        end
        
        bar(i,mean(latIdx(:,i,iPlot),1),...
            'FaceColor',thisColor); hold on
        
    end
    
    errorbar(mean(latIdx(:,:,iPlot),1),std(latIdx(:,:,iPlot),0,1)/sqrt(size(latIdx,1)),...
        'Color','k',...
        'LineStyle','none',...
        'LineWidth',2.5,...
        'CapSize',0)
    
    set(gca,...
        'ylim',[0,0.09],...
        'YTick',-.04:.04:.16,...
        'xlim',[.5,2.5],...
        'xTickLabels',{' ', ' '},...
        'xTick',[],...
        'box','off',...
        'linewidth',1.5,...
        'fontsize',16)
    
    title(thisTimeSegmentLabel);
    
    pbaspect([1,1,1])
end

save([sourceDir 'Alpha_lateralization.mat'],'latIdx','-v7.3');


% convert lat index to wide format for R
% for iData=1:2
%
%     % get data from time segment
%     theseData = squeeze(latIdx(:,:,iData));
%
%     % convert into R "long" format
%     nSubs = size(theseData,1);
%     dumCon = [zeros(nSubs,1), ones(nSubs,1)];
%
%     % create column vector of subjects
%     sjNums = 1:size(theseData,1);
%     sjNums = [sjNums, sjNums]';
%
%     % use colon operator function to get into long format
%     if iData==1
%         stimPeriodData_LF = [theseData(:), sjNums, dumMem(:), dumEyes(:)];
%     elseif iData==2
%         RetData_LF = [theseData(:), sjNums, dumMem(:), dumEyes(:)];
%     end
%
% end

% save data
%save([sourceDir '/' 'Alpha_Lateralization_Data.mat'],'latIdx','stimPeriodData_LF','RetData_LF','-v7.3')

% save plot
%saveas(h,[plotDir '/' 'Alpha_Lateralization_Plots.eps'],'epsc')



% % run a quick ANOVA (just double checking null BF results)
% addpath(genpath('/home/waldrop/Desktop/WTF_EYE/resampling'))
% % name variables
% var1_name = 'eyes';
% var1_levels = 2;
% var2_name = 'mem';
% var2_levels = 2;
%
% for i=1:3
%
%     observedData = squeeze(latIdx(:,:,i));
%
%     statOutput = teg_repeated_measures_ANOVA(observedData,[var1_levels var2_levels],{var1_name, var2_name});
%
%     allAnova(:,:,i) = statOutput;
%
% end











%
%
%
%             subplot(100,100,100)
%
%             %% NOTE THAT EACH LOCATION IS INDEPENDENTLY NORMALIZED
%
%             % find max and min across all locations for this condition/time segment
%             thisMax = max(squeeze(mean(mean(mean(allBand(:,iCond,iLoc,theseChans,thisTimeSegment)),5),1)));
%             thisMin = min(squeeze(mean(mean(mean(allBand(:,iCond,iLoc,theseChans,thisTimeSegment)),5),1)));
%
%             % normalize the bandpassed data
%             allBandNorm = (allBand-thisMin)/(thisMax-thisMin);
%
%             % set data
%             theseData = squeeze(mean(mean(allBandNorm(:,iCond,iLoc,:,thisTimeSegment),1),5));
%
%             %headplot(theseData,'wtfSpline.spl','view','back','maplimits',theseMapLimits,'electrodes',showElects) %'cbar',0,
%             ax(iLoc)=headplot(theseData,'wtfSpline.spl','view','back','electrodes',showElects,'maplimits',theseMapLimits) %'cbar',0,
%
%             % set placement for each condition's topos
%             if iCond==1
%                 centerX=.3;
%                 centerY=.6;
%             elseif iCond==2
%                 centerX=.6;
%                 centerY=.6;
%             elseif iCond==3
%                 centerX=.3;
%                 centerY=.15;
%             elseif iCond==4
%                 centerX=.6;
%                 centerY=.15;
%             end
%
%             % set figure sizes
%             sizeX=.15;
%             sizeY=.15;
%
%             if      iLoc==1; xyShift = [.08,.07];
%             elseif  iLoc==2; xyShift = [.03,.14];
%             elseif  iLoc==3; xyShift = [-.03,.14];
%             elseif  iLoc==4; xyShift = [-.08,.07];
%             elseif  iLoc==5; xyShift = [-.08,-.07];
%             elseif  iLoc==6; xyShift = [-.03,-.14];
%             elseif  iLoc==7; xyShift = [.03,-.14];
%             elseif  iLoc==8; xyShift = [.08,-.07];
%             end
%
%             ax(iLoc).Position = [centerX+xyShift(1),centerY+xyShift(2),sizeX,sizeY];
%
%             % add title to each plot
%             %title(['Loc ' num2str(iLoc)],'FontSize',24)
%
%         end
%
%         % add condition labels to plot
%         if      iCond==1; condTitle = 'S/F';
%         elseif  iCond==2; condTitle = 'C/F';
%         elseif  iCond==3; condTitle = 'S/M';
%         elseif  iCond==4; condTitle = 'C/M';
%         end
%         annotation('textbox', [centerX+.04,centerY+.11, 0,0], 'string', condTitle,'FontSize',36);
%
%         clear allBandNorm
%
%     end
%
%     % add title
%     if timeSegmentToPlot==1
%         figureTitlePosition=[.25,.97,0,0];
%     elseif timeSegmentToPlot==2
%         figureTitlePosition=[.25,.97,0,0];
%     elseif timeSegmentToPlot==3
%         figureTitlePosition=[.25,.97,0,0];
%     end
%     annotation('textbox',figureTitlePosition, 'string', thisTimeSegmentLabel,'FontSize',36,'FitBoxToText','on','LineStyle','none');
%
%     if timeSegmentToPlot==1
%         saveas(h,[plotDir '/' 'Alpha_Topos_' thisTimeSegmentTitle '.png'],'png')
%     elseif timeSegmentToPlot==2
%         saveas(h,[plotDir '/' 'Alpha_Topos_' thisTimeSegmentTitle '.png'],'png')
%     elseif timeSegmentToPlot==3
%         saveas(h,[plotDir '/' 'Alpha_Topos_' thisTimeSegmentTitle '.png'],'png')
%     end
%
%     clear ax theseData allBandNorm
%
%
% end
end