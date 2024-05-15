function TrainBoth_computeSlopes_All_singleFreqs_nonParam (sub)
%==========================================================================
%{
calculateSlopes_Single_Freqs
Purpose: calculate slope values for single freq bandpass IEM analyses

Original Author:
Joshua J. Foster
joshua.james.foster@gmail.com
University of Chicago
August 17, 2015

Modified by Tom Bullock
UCSB Attention Lab
%}
%==========================================================================

close all

% setup directories
root = '/home/garrett/WTF_Bike/'; %out = 'AnalysisScripts/trunk/MATLAB';
dRoot = [root,'Data/TrainBoth/'];


name = '_SpatialTF_allFreq.mat'; % name of files to be saved

saveName = '_CRFslopes_allFreq.mat';


%matlabpool open 72


% clear stuff before next loop
em = []; pDat_rest = []; pSl_rest = []; rDat_rest = []; rSl_rest = []; d= []; dat = [];rPw_rest=[]; pPw_rest=[];

pDat_ex = []; pSl_ex = []; rDat_ex = []; rSl_ex = []; rPw_ex = []; pPw_ex = [];

% grab subject's data (*_SpatialTF_Permed also contains real DTFs)
fName = [dRoot,sprintf('sj%02d_TrainBoth_Permute_changeDect_accTrials_',sub), name];
tmp = load(fName);
em = tmp.em;
tmp.em = [];

rDat_rest.total = em.tfs_rest.total;
pDat_rest.total = squeeze(em.permtfs.tfs_rest.total); % freqs x 1 x blocks x time x channels

rDat_ex.total = em.tfs_low.total;
pDat_ex.total = squeeze(em.permtfs.tfs_low.total);

% Specify properties
nChans = em.nChans; % # of location channels
nPerms = size(pDat_rest.total,2); % % of permutations
%nSamps = em.dSamps; % # of samples (after downsampling)
nFreqs = length(em.frequencies)-1; % # of frequencies
nSamps = size(rDat_rest.total,2); %Jordan changed to 2 for 625 samples


% SPECIFY X-values (TOM ADDITION)
%thisX = 1:5; % foster uses this
x = 0:45:180; % WTF use real angular values

% % real evoked data
% for f = 1:nFreqs
%     for samp = 1:nSamps
%         dat = squeeze(rDat_rest.evoked(f,samp,:));
%         %dat = squeeze(rDat.evoked(f,samp,:));
%         x = thisX;
%         d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
%         fit = polyfit(x,d,1);
%         rSl.evoked(f,samp)= fit(1);
%         rPw.evoked(f,samp)=dat(5);
%     end
% end

% Rest real total data
for f = 1:nFreqs
    for samp = 1:nSamps
        dat = squeeze(rDat_rest.total(f,samp,:));
        %dat = squeeze(rDat.evoked(f,samp,:));
        d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
        fit = polyfit(x,d,1);
        rSl_rest.total(f,samp)= fit(1);
        rPw_rest.total(f,samp)=dat(5);
    end
end

% Ex real total data
for f = 1:nFreqs
    for samp = 1:nSamps
        dat = squeeze(rDat_ex.total(f,samp,:));
        %dat = squeeze(rDat.evoked(f,samp,:));
        d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
        fit = polyfit(x,d,1);
        rSl_ex.total(f,samp)= fit(1);
        rPw_ex.total(f,samp)=dat(5);
    end
end


% permuted evoked data
% for perm = 1:nPerms
%     disp(['Evoked Perm: ' num2str(perm)])
%     for f = 1:nFreqs
%         for samp = 1:nSamps
%             dat = squeeze(pDat_rest.evoked(f,perm,samp,:));
%             %dat = squeeze(pDat.evoked(f,perm,samp,:));
%             x = thisX;
%             d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
%             fit = polyfit(x,d,1);
%             pSl.evoked(f,samp,perm)= fit(1) ;
%             pPw.evoked(f,samp,perm)=dat(5);
%         end
%     end
% end

% permuted Rest total data
for perm = 1:nPerms
    for f = 1:nFreqs
        for samp = 1:nSamps
            dat = squeeze(pDat_rest.total(f,perm,samp,:));
            d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
            fit = polyfit(x,d,1);
            pSl_rest.total(f,samp,perm)= fit(1) ;
            pPw_rest.total(f,samp,perm)=dat(5);
        end
    end
end

% permuted Ex total data
for perm = 1:nPerms
    for f = 1:nFreqs
        for samp = 1:nSamps
            dat = squeeze(pDat_ex.total(f,perm,samp,:));
            d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
            fit = polyfit(x,d,1);
            pSl_ex.total(f,samp,perm)= fit(1) ;
            pPw_ex.total(f,samp,perm)=dat(5);
        end
    end
end


% save slope matrices
filename = [dRoot,sprintf('sj%02d_TrainBoth_changeDect_accTrials',sub),saveName];
save(filename,'rSl_rest','pSl_rest','rSl_ex','pSl_ex','-v7.3');




clear all
close all

end