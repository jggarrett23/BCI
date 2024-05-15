function slopes_nonParm_clustering_mcc (band,wave,model_type)
%{
=========================================================================
Performs permutation and applies a correction for multiple comparisons
based off of clustering. Conducts both frequentist T-test and BF T-tests.

Author: Jordan Garrett
jordangarrett@ucsb.edu
Attention Lab
=========================================================================

band: 1 = Alpha, 2 = Theta
model_type: 1 = Independent, 2 = Fixed
%}

parentDir = '/work/garrett/WTF_Bike/';

if wave == 1
    wave_type = 'total';
else
    wave_type = 'evoked';
end


if band == 1
    freq_name = 'ALPHA';
else
    freq_name = 'THETA';
end

if model_type == 1
    dataDir = [parentDir 'Data_Compiled/'];
    file_suffix = sprintf('_FreqStat_Slopes_%s_%s_accTrials',wave_type,freq_name);
    model_saveName = 'Indep'; 
else
    dataDir = [parentDir 'Data/TrainBoth/Data_Compiled/'];
    file_suffix = sprintf('_TrainBoth_FreqStat_Slopes_%s_%s_accTrials',wave_type,freq_name);
    model_saveName = 'Fixed';
end

% Load data
tmp_rest = load([dataDir,'exerCon01_changeDect' file_suffix]);

tmp_ex = load([dataDir,'exerCon02_changeDect' file_suffix]);

if wave_type == 1
    allRest_real = tmp_rest.allTF_real_total;
    allRest_perm = tmp_rest.allTF_perm_total;

    allEx_real = tmp_ex.allTF_real_total;
    allEx_perm = tmp_ex.allTF_perm_total;
else
    allRest_real = tmp_rest.allTF_real_evoked;
    allRest_perm = tmp_rest.allTF_perm_evoked;

    allEx_real = tmp_ex.allTF_real_evoked;
    allEx_perm = tmp_ex.allTF_perm_evoked;
end


time = linspace(-500,2000,625);

%% Real Slope Stats

% T-Tests

% Do one sample t-test comparing real slopes to a mean of 0. Save t-values
[h,p,ci,restReal_stat] = ttest(allRest_real);
restReal_t = restReal_stat.tstat;

[h,p,ci,exReal_stat] = ttest(allEx_real);
exReal_t = exReal_stat.tstat;

[h,p,ci,pairedReal_stat] = ttest(allRest_real,allEx_real,'Dim',1);
paired_t = pairedReal_stat.tstat;

% BF T-Tests
rest_realBF = []; 
ex_realBF = [];
restEx_realBF = [];

for iSamp = 1:size(allRest_real,2)
   
    %One sample BF T-test Rest
    [b10_rest,p_rest] = bf.ttest(allRest_real(:,iSamp));
    rest_realBF(iSamp) = b10_rest;
    
    
    %One sample BF T-test Exercise
    [b10_ex,p_ex] = bf.ttest(allEx_real(:,iSamp));
    ex_realBF(iSamp) = b10_ex;
    
    %Paired samples BF T-test Rest v Exercise
    [b10_restEx,p_restEx] = bf.ttest(allRest_real(:,iSamp),allEx_real(:,iSamp));
    restEx_realBF(iSamp) = b10_restEx;
    
end
%% Perm Slope Stats

%Loop through each iteration and perform one-sample t-test for each time
%point of permuted slopes

allRest_perm = permute(allRest_perm,[3,1,2]);
allEx_perm = permute(allEx_perm,[3,1,2]);

restPerm_tDist = zeros(size(allRest_perm,1),size(allRest_perm,3));
exPerm_tDist = restPerm_tDist;
rest_ex_Perm_tDist = restPerm_tDist;

restPerm_bfDist = restPerm_tDist;
exPerm_bfDist = restPerm_bfDist;
rest_ex_Perm_bfDist = exPerm_bfDist;

for iIter = 1:size(allRest_perm,1)
    
    restCurrent_it = squeeze(allRest_perm(iIter,:,:));
    exCurrent_it = squeeze(allEx_perm(iIter,:,:));
    
    % running t-test on a matrix will perform a one-sample t-test for each
    % column of the matrix
    [rest_h,rest_p,rest_ci,rest_stats] = ttest(restCurrent_it);
    
    [ex_h,ex_p,ex_ci,ex_stats] = ttest(exCurrent_it);
    
    restT_values = rest_stats.tstat;
    exT_values = ex_stats.tstat;
    
    restPerm_tDist(iIter,:) = restT_values;
    exPerm_tDist(iIter,:) = exT_values;
    
    % do same for paired-samples to get distribution of paired sample
    % t-statistics
    [paired_h,paired_p,paired_ci,paired_stats] = ttest(restCurrent_it,exCurrent_it);
    pairedT_values = paired_stats.tstat;
    rest_ex_Perm_tDist(iIter,:) = pairedT_values;
    
    
    % Bayes has to be ran on each time point separately
    for iSamp = 1:size(allRest_real,2)
       
        %Rest
        [b10_rest,p_rest] = bf.ttest(allRest_perm(iIter,:,iSamp));
        restPerm_bfDist(iIter,iSamp) = b10_rest;
       
        %Exercise
        [b10_ex,p_ex] = bf.ttest(allEx_perm(iIter,:,iSamp));
        exPerm_bfDist(iIter,iSamp) = b10_ex;
        
        %Rest v Exercise
        [b10_restEx,p_restEx] = bf.ttest(allRest_perm(iIter,:,iSamp),allEx_perm(iIter,:,iSamp));
        rest_ex_Perm_bfDist(iIter,iSamp) = b10_restEx;
        
    end
    
end

% save null distributions
slopes.nonParam_stats.BF.Nulls.Rest = restPerm_bfDist;
slopes.nonParam_stats.BF.Nulls.Ex = exPerm_bfDist;
slopes.nonParam_stats.BF.Nulls.restEx = rest_ex_Perm_bfDist;

%% Compare Real T-value to Permuted Distribution
alpha = 0.05;
cut_off = (1-alpha)*100;

% Loop through and determine if real statistic is greater than 1-alpha of
% the permuted t-distribution

rest_sigTime = zeros(1,length(restReal_t));
ex_sigTime = zeros(1,length(exReal_t));
paired_sigTime = rest_sigTime;

for iSamp = 1:length(restReal_t)
   
    %get current t-value associated with 1-alpha percentile
    rest_tThresh = prctile(restPerm_tDist(:,iSamp),cut_off);
    
    ex_tThresh = prctile(exPerm_tDist(:,iSamp),cut_off);
    
    rest_observedT = restReal_t(iSamp);
    ex_observedT = exReal_t(iSamp);
    
    if rest_observedT > rest_tThresh
        rest_sigTime(iSamp) = 1;
    end
    
    if ex_observedT > ex_tThresh
        ex_sigTime(iSamp) = 1;
    end
    
    % paired samples comparisons
    paired_tThresh = prctile(rest_ex_Perm_tDist(:,iSamp),cut_off);
    paired_observedT = paired_t(iSamp);
    
    if paired_observedT > paired_tThresh
        paired_sigTime(iSamp) = 1;
    end
    
end

%% Cohen's method (Ch 33)

% For Cohen's approach I dont believe that we compare our statistic to a
% permuted distribution of that statistic. Rather, we get the max cluster
% size of permuted slopes, then compare the cluster size of real slopes to
% permuted distribution of max cluster sizes

cluster_thresh = 0.05;

restPerm_bfThresh = restPerm_bfDist;
restPerm_bfThresh(restPerm_bfDist < 3 ) = 0;

exPerm_bfThresh = exPerm_bfDist;
exPerm_bfThresh(exPerm_bfDist < 3 ) = 0;

restEx_perm_bfThresh = rest_ex_Perm_bfDist;
restEx_perm_bfThresh(rest_ex_Perm_bfDist < 3 ) = 0;

restPerm_bf_maxCluster = zeros(1,size(restPerm_bfThresh,1));
exPerm_bf_maxCluster = restPerm_bf_maxCluster;

restEx_permBF_maxCluster = restPerm_bf_maxCluster;
for iIter = 1:size(restPerm_bfThresh,1)
    
    %Rest
    % this function finds continuous non-zero elements
    restPerm_clustInfo = bwconncomp(restPerm_bfThresh(iIter,:));
    
    % over all the clusters, calcuate there size and store the maximum one
    restPerm_bf_maxCluster(iIter) = max([ 0 cellfun(@numel,restPerm_clustInfo.PixelIdxList)]);
    
    %Exercise
    exPerm_clustInfo = bwconncomp(exPerm_bfThresh(iIter,:));
    exPerm_bf_maxCluster(iIter) = max([ 0 cellfun(@numel,exPerm_clustInfo.PixelIdxList)]);
    
    %Rest v Exercise
    restEx_Perm_clustInfo = bwconncomp(restEx_perm_bfThresh(iIter,:));
    restEx_permBF_maxCluster(iIter) = max([ 0 cellfun(@numel,restEx_Perm_clustInfo.PixelIdxList)]);
    
end

% Now threshold the real slopes, and then keep clusters that are larger
% a percentage (e.g. 95% of the max cluster distribution)
rest_realBF_thresh = rest_realBF;
rest_realBF_thresh(rest_realBF < 3) = 0;

ex_realBF_thresh = ex_realBF;
ex_realBF_thresh(ex_realBF < 3) = 0;

restEx_realBF_thresh = restEx_realBF;
restEx_realBF_thresh(restEx_realBF < 3) = 0;

restReal_bfClusters = bwconncomp(rest_realBF_thresh);
restReal_bfClust_info = cellfun(@numel,restReal_bfClusters.PixelIdxList);
restReal_bfClust_thresh = prctile(restPerm_bf_maxCluster,100-cluster_thresh*100);
restCluster_remove = find(restReal_bfClust_info <= restReal_bfClust_thresh);

exReal_bfClusters = bwconncomp(ex_realBF_thresh);
exReal_bfClust_info = cellfun(@numel,exReal_bfClusters.PixelIdxList);
exReal_bfClust_thresh = prctile(exPerm_bf_maxCluster,100-cluster_thresh*100);
exCluster_remove = find(exReal_bfClust_info <= exReal_bfClust_thresh);

restEx_realBF_clusters = bwconncomp(restEx_realBF_thresh);
restEx_realBF_clust_Info = cellfun(@numel,restEx_realBF_clusters.PixelIdxList);
restEx_realBFClust_thresh = prctile(restEx_permBF_maxCluster,100-cluster_thresh*100);
restEx_clusterRemove = find(restEx_realBF_clust_Info <= restEx_realBFClust_thresh);

%Remove clusters

%Rest
for iClust = 1:length(restCluster_remove)
    rest_realBF_thresh(restReal_bfClusters.PixelIdxList{restCluster_remove(iClust)})=0;
end

%Exercise
for iClust = 1:length(exCluster_remove)
    ex_realBF_thresh(exReal_bfClusters.PixelIdxList{exCluster_remove(iClust)})=0;
end

%Rest v Ex
for iClust = 1:length(restEx_clusterRemove)
    restEx_realBF_thresh(restEx_realBF_clusters.PixelIdxList{restEx_clusterRemove(iClust)})=0;
end

%% Save statistics

% Save significant time points according to T
slopes.nonParam_stats.T.RestvPerm = rest_sigTime;
slopes.nonParam_stats.T.ExvPerm = ex_sigTime;
slopes.nonParam_stats.T.RestvEx = paired_sigTime;

% Save BF stats
slopes.nonParam_stats.BF.Rest.Real = rest_realBF;
slopes.nonParam_stats.BF.Ex.Real = ex_realBF;
slopes.nonParam_stats.BF.RestEx.Real = restEx_realBF;

slopes.nonParam_stats.BF.corrected.RestvPerm = rest_realBF_thresh;
slopes.nonParam_stats.BF.corrected.ExvPerm = ex_realBF_thresh;
slopes.nonParam_stats.BF.corrected.RestvEx = restEx_realBF_thresh;

%average over slope iterations and save here
slopes.Rest_permSlope = squeeze(mean(allRest_perm,1));
slopes.Ex_permSlope = squeeze(mean(allEx_perm,1));

slopes.allRest_realSlope = allRest_real;
slopes.allEx_realSlope = allEx_real;

slopes.avgRest_realSlope = mean(allRest_real,1);
slopes.avgEx_realSlope = mean(allEx_real,1);

slopes.Rest_realError = squeeze(std(allRest_real,0,1)/sqrt(size(allRest_real,1)));
slopes.Ex_realError = squeeze(std(allEx_real,0,1)/sqrt(size(allEx_real,1)));

slopes.time = time;

save([dataDir,...
    sprintf('%s_%s_%s_nonParam_SlopeStats.mat',model_saveName,freq_name,wave_type)], 'slopes')

end