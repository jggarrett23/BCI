function computeTrainBoth_genStats

%% Load Data and Set Analysis parameters
parentDir = '/home/garrett/WTF_Bike/';
dataDir = [parentDir 'Data/TrainBoth/Data_Compiled/'];
saveDir = dataDir;

% load data
load([dataDir 'TrainBoth_Gen_CTFslopes_ALPHA.mat'],'allTF_real_total','allTF_perm_total');

rest_realGen = allTF_real_total.rest;
ex_realGen = allTF_real_total.ex;

rest_permGen = allTF_perm_total.rest;
ex_permGen = allTF_perm_total.ex;

nIter = size(rest_permGen,2);
nSamps = size(rest_realGen,2);
cluster_alpha = 0.05; % for cluster correction thresholds

%% Statistical Tests

%================== BF T-tests ======================

% Real Generalizations
rest_realGen_BF = nan(nSamps,nSamps);
ex_realGen_BF = rest_realGen_BF;
restEx_realGen_BF = rest_realGen_BF;
for trSamp = 1:nSamps
    for teSamp = 1:nSamps
        
        % One Sample
        rest_realGen_BF(trSamp,teSamp) = bf.ttest(rest_realGen(:,trSamp,teSamp));
        ex_realGen_BF(trSamp,teSamp) = bf.ttest(ex_realGen(:,trSamp,teSamp));
        
        % Paired Samples
        restEx_realGen_BF(trSamp,teSamp) = bf.ttest(rest_realGen(:,trSamp,teSamp),...
            ex_realGen(:,trSamp,teSamp));
    end
end

% Permuted Generalizations
rest_permGen_BF = nan(nIter,nSamps,nSamps);
ex_permGen_BF = rest_permGen_BF;
restEx_permGen_BF = rest_permGen_BF;
for iIter = 1:nIter
    for trSamp = 1:nSamps
        for teSamp = 1:nSamps
            
            rest_permIt = squeeze(rest_permGen(:,iIter,trSamp,teSamp));
            ex_permIt = squeeze(ex_permGen(:,iIter,trSamp,teSamp));
            
            % One Sample
            rest_permGen_BF(iIter,trSamp,teSamp) = bf.ttest(rest_permIt);
            ex_permGen_BF(iIter,trSamp,teSamp) = bf.ttest(ex_permIt);
            
            % Paired Samples
            restEx_permGen_BF(iIter,trSamp,teSamp) = bf.ttest(rest_permIt,ex_permIt);
        end
    end
end

% Threshold BF matrices
restReal_genBF_thresh = rest_realGen_BF;
restReal_genBF_thresh(rest_realGen_BF < 3) = 0;

exReal_genBF_thresh = ex_realGen_BF;
exReal_genBF_thresh(ex_realGen_BF < 3) = 0;

restEx_realGen_BFthresh = restEx_realGen_BF;
restEx_realGen_BFthresh(restEx_realGen_BF < 3) = 0;

restPerm_genBF_thresh = rest_permGen_BF;
restPerm_genBF_thresh(rest_permGen_BF < 3) = 0;

exPerm_genBF_thresh = ex_permGen_BF;
exPerm_genBF_thresh(ex_permGen_BF < 3) = 0;

restEx_permGen_BFthresh = restEx_permGen_BF;
restEx_permGen_BFthresh(restEx_permGen_BF < 3) = 0;

%% Cluster Correction

% Max cluster size distributions of contiguous BF for permuted slopes 
restReal_genMax_permClustDist = nan(1,nIter);
exReal_genMax_permClustDist = restReal_genMax_permClustDist;
restEx_realGen_maxPerm_clustDist = restReal_genMax_permClustDist;

for iIter = 1:nIter
    
    % Rest
    restEx_currentMat = squeeze(restPerm_genBF_thresh(iIter,:,:));
    clustInfo = bwconncomp(restEx_currentMat);
    maxClust_info = max([ 0 cellfun(@numel,clustInfo.PixelIdxList) ]);
    restReal_genMax_permClustDist(iIter) = maxClust_info;
    
    % Exercise
    ex_currentMat = squeeze(exPerm_genBF_thresh(iIter,:,:));
    clustInfo = bwconncomp(ex_currentMat);
    maxClust_info = max([ 0 cellfun(@numel,clustInfo.PixelIdxList) ]);
    exReal_genMax_permClustDist(iIter) = maxClust_info;
    
    % Rest vs Exercise
    restEx_currentMat = squeeze(restEx_permGen_BFthresh(iIter,:,:));
    clustInfo = bwconncomp(restEx_currentMat);
    maxClust_info = max([ 0 cellfun(@numel,clustInfo.PixelIdxList) ]);
    restEx_realGen_maxPerm_clustDist(iIter) = maxClust_info;
end

% Remove "non-sig" BF clusters according to cluster threshold using max cluster
% size distribution of permuted BFs

%---------- Rest --------------------------
thisBF_mat = restReal_genBF_thresh;
thisBF_permDist = restReal_genMax_permClustDist;

clustInfo = bwconncomp(thisBF_mat); %clusters of Real BFs

clustSizes = cellfun(@numel,clustInfo.PixelIdxList); % sizes of clusters

clustThresh = prctile(thisBF_permDist ,100-cluster_alpha*100);

%compare to perm dist clust sizes and find clusters to remove
nonSig_clusts = find(clustSizes < clustThresh);

for iClust = 1:length(nonSig_clusts)
    thisBF_mat(clustInfo.PixelIdxList{nonSig_clusts(iClust)}) = 0;
end

restReal_genBF_threshCC = thisBF_mat;


%----------- Ex ----------------------------
thisBF_mat = exReal_genBF_thresh;
thisBF_permDist = exReal_genMax_permClustDist;

clustInfo = bwconncomp(thisBF_mat); %clusters of Real BFs

clustSizes = cellfun(@numel,clustInfo.PixelIdxList); % sizes of clusters

clustThresh = prctile(thisBF_permDist,100-cluster_alpha*100);

%compare to perm dist clust sizes and find clusters to remove
nonSig_clusts = find(clustSizes < clustThresh);

for iClust = 1:length(nonSig_clusts)
    thisBF_mat(clustInfo.PixelIdxList{nonSig_clusts(iClust)}) = 0;
end

exReal_genBF_threshCC = thisBF_mat;


%----------- Rest v Ex ---------------------
thisBF_mat = restEx_realGen_BFthresh;
thisBF_permDist = restEx_realGen_maxPerm_clustDist;

clustInfo = bwconncomp(thisBF_mat); %clusters of Real BFs

clustSizes = cellfun(@numel,clustInfo.PixelIdxList); % sizes of clusters

clustThresh = prctile(thisBF_permDist,100-cluster_alpha*100);

%compare to perm dist clust sizes and find clusters to remove
nonSig_clusts = find(clustSizes < clustThresh);

for iClust = 1:length(nonSig_clusts)
    thisBF_mat(clustInfo.PixelIdxList{nonSig_clusts(iClust)}) = 0;
end

restEx_realGen_BFthreshCC = thisBF_mat;

%% Save data

% Uncorrected BF
genStats.real.rawBF.rest = rest_realGen_BF;
genStats.real.rawBF.ex = ex_realGen_BF;
genStats.real.rawBF.restEx = restEx_realGen_BF;

% Thresholded Clusters
genStats.real.threshCC.rest = restReal_genBF_threshCC;
genStats.real.threshCC.ex = exReal_genBF_threshCC;
genStats.real.threshCC.restEx = restEx_realGen_BFthreshCC;

% Average Generalizations
genStats.avgGen.rest = squeeze(mean(rest_realGen));
genStats.avgGen.ex = squeeze(mean(ex_realGen));

save([saveDir 'TrainBoth_genStats.mat'],'genStats','-v7.3');
end