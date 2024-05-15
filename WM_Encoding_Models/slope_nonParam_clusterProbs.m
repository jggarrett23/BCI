function slope_nonParam_clusterProbs

%{
==========================================================
Compute probabilities of getting Bayes Factor >= 3, then computes 
probability of getting cluster sizes.
==========================================================
%}

parentDir = '/home/garrett/WTF_Bike/';
dataDir = [parentDir 'Data/TrainBoth/Data_Compiled/'];
saveDir = [parentDir 'Overall_Images/'];


% load in data
load([dataDir 'Fixed_ALPHA_nonParam_SlopeStats.mat'],'slopes');

restBF_nullDist = slopes.nonParam_stats.BF.Nulls.Rest;
exBF_nullDist = slopes.nonParam_stats.BF.Nulls.Ex;
restEx_bfNull_Dist = slopes.nonParam_stats.BF.Nulls.restEx;

% Compute the Probability of getting a Bayes Factor of >= 3
rest_null = reshape(restBF_nullDist,1,[]);
ex_null = reshape(exBF_nullDist,1,[]);
restEx_null = reshape(restEx_bfNull_Dist,1,[]);

restBF_prob = (sum(rest_null >= 3))/length(rest_null);
exBF_prob = (sum(ex_null >= 3))/length(ex_null);
restEx_prob = (sum(restEx_null >= 3))/length(restEx_null);

fprintf('P(BF >= 3|Rest) = %.3f \n',restBF_prob)
fprintf('P(BF >= 3|Exercise) = %.3f \n',exBF_prob)
fprintf('P(BF >= 3|Rest v Ex) = %.3f \n',restEx_prob)

% Compute cluster distribution for BF >= 3
wholeRest_null = round(restBF_nullDist);
wholeEx_null = round(exBF_nullDist);
whole_restEx_Null = round(restEx_bfNull_Dist);

% Only look at BF from 1-10 and greater
rest_clusters = {};
ex_clusters = {};
restEx_clusters = {};

bF_oi = 3:10;

for iBF = 1:(length(bF_oi)+1) % add one here to group BF > 10 together
    
    if iBF <= length(bF_oi)
        thisBF = bF_oi(iBF);
    end
    
    for iIter = 1:size(wholeRest_null,1)
        
        if iBF <= length(bF_oi)
            thisBF_restIdx = wholeRest_null(iIter,:)==thisBF;
            thisBF_exIdx = wholeEx_null(iIter,:)==thisBF;
            thisBF_restEx_Idx = whole_restEx_Null(iIter,:)==thisBF;
        else %group BF > 10 together
            thisBF_restIdx = wholeRest_null(iIter,:) > 10;
            thisBF_exIdx = wholeEx_null(iIter,:) > 10;
            thisBF_restEx_Idx = whole_restEx_Null(iIter,:) > 10;
        end
        
        rest_clustInfo = bwconncomp(thisBF_restIdx);
        %save the size of clusters for each iteration
        rest_clusters{iBF,iIter} = [ 0 cellfun(@numel,rest_clustInfo.PixelIdxList)];
        
        ex_clustInfo = bwconncomp(thisBF_exIdx);
        ex_clusters{iBF,iIter} = [ 0 cellfun(@numel,ex_clustInfo.PixelIdxList)];
        
        restEx_clustInfo = bwconncomp(thisBF_restEx_Idx);
        restEx_clusters{iBF,iIter} = [ 0 cellfun(@numel,restEx_clustInfo.PixelIdxList)];
        
    end
    
end

% Create Frequency Distribution of BF clusters
all_clustDist = {};
for iBF = 1:size(rest_clusters,1)
    
    % Log BFs
    if iBF <= length(bF_oi)
        all_clustDist{iBF,1} = ['BF = ' int2str(bF_oi(iBF))];
    else
        all_clustDist{iBF,1} = 'BF > 10';
    end
    
    %Rest
    restClust_freqTab= tabulate(horzcat(rest_clusters{iBF,:}));
    all_clustDist{iBF,2}.Sizes = restClust_freqTab(:,1);
    all_clustDist{iBF,2}.Freqs = restClust_freqTab(:,2);
    
    % Freq/[N_iterations*(N_timepoints - ClusterSize) + 1]
    all_clustDist{iBF,2}.Probs = restClust_freqTab(:,2)./(size(rest_clusters,2)*(625-restClust_freqTab(:,1))+1);
    
    % Ex
    exClust_freqTab= tabulate(horzcat(ex_clusters{iBF,:}));
    all_clustDist{iBF,3}.Sizes = exClust_freqTab(:,1);
    all_clustDist{iBF,3}.Freqs = exClust_freqTab(:,2);
    all_clustDist{iBF,3}.Probs = exClust_freqTab(:,2)./(size(ex_clusters,2)*(625-exClust_freqTab(:,1))+1);
    
    % RestvEx
    restEx_Clust_freqTab= tabulate(horzcat(restEx_clusters{iBF,:}));
    all_clustDist{iBF,4}.Sizes = restEx_Clust_freqTab(:,1);
    all_clustDist{iBF,4}.Freqs = restEx_Clust_freqTab(:,2);
    all_clustDist{iBF,4}.Probs = restEx_Clust_freqTab(:,2)./(size(restEx_clusters,2)*(625-restEx_Clust_freqTab(:,1))+1);
    
end


% Plotting
clusterProbs_fig = figure('units','inches');
line_colors = parula(10);
for iCompare = 1:3
    
    if iCompare == 1
        titleName = 'Rest';
    elseif iCompare == 2
        titleName = 'Exercise';
    else
        titleName = 'Rest v Exercise';
    end
    
    compare_colm = iCompare + 1;
    
    subplot(1,3,iCompare)
    for iBF = 1:size(all_clustDist,1)
        
        thisSizes = all_clustDist{iBF,compare_colm}.Sizes;
        thisFreq = all_clustDist{iBF,compare_colm}.Freqs;
        thisProb = all_clustDist{iBF,compare_colm}.Probs;
        
        % only plot sizes greater than 2
        k_idx = thisSizes >= 2;
        
        
        plot(thisSizes(k_idx),thisProb(k_idx),'LineWidth',2,'Color',line_colors(iBF,:)); hold on
        
    end
    
    set(gca,'fontsize',10,'fontweight','bold','linewidth',2)
    
    if iCompare == 3
        leg = legend(all_clustDist{:,1},'fontsize',7.5,'linewidth',1,'fontweight','normal');
        legend('boxoff')
        leg.Position = [0.7836,0.2962,0.1298,0.5958];
    end
    box off
    
    if iCompare == 1
        ylabel('Probability')
    end
    
    xlabel('Cluster size')
    xlim([2,20])
    tick_pos = [2,10,20];
    xticks(tick_pos)
    t = title(titleName,'fontsize',10,'fontweight','bold','Position',[11,.0016,0]);
    
    if iCompare == 3
       t.Position = [14,.0016,0]; 
    end
    
end

set(clusterProbs_fig,'Position',[3.4750,1.2333,7.193,2])
%sgtitle({'Fixed Model Probability of';'Continuous Bayes Factor Cluster'},'fontsize',15,'fontweight','bold')

saveas(clusterProbs_fig,[saveDir 'Fixed_Alpha_nonParam_bfCluster_probsPlots.jpg']);
close all;

end