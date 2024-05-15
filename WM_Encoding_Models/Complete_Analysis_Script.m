%========================================================================================================
%{ 
This is the order in which the data is processed after running a subject.
It is important though to make sure that individual an subject's data is
inspected after running this script. 
Author: Jordan Garrett
%}
%=========================================================================================================
%% Complete Data Analysis Script

analysis_dir =  '/work/garrett/WTF_Bike/Analysis_Scripts';
cd (analysis_dir)

%change subjects as you recruit more
subjects=[35]; %for individual subject processing
total_subs = [1:8,10:35]; %for overall processing, necessary for some functions

% rest, low
conditions =[1:2];

plotting = 1;

%correct trials only settings
accuracy = [0:1];
acc = 0;

%need to check number of trials rejected?
check_trials = 0; % ON = 1, OFF= 0

%post exercise effects testing
post_ex = 1; % 0 = OFF
low_first = [3,5,6,7,8,11,12,13,16,17,23,25,29,33,34,35]; %subjects who did condition order 2,1
rest_first = [1,2,4,10,14,15,19,20,21,22,24,26,27,28,30,31,32]; % 1,2

%% EEG CLEANING

% Merge behavior data
Beh_Merge_Blocks (subjects, conditions)


cd(analysis_dir)
preProc_job1 = EEG_preprocessing1_job (subjects);

% pay attention to number of trials rejected here
cd(analysis_dir)
if check_trials == 0
    preProc_job2 = EEG_preprocessing2_job(subjects);
else
    %second value controls condition, last one checks the topo
    EEG_preprocessing2(subjects,1,1,1,0)
    cd(analysis_dir)
    return
end


%% IEM
try
    %needed for IEM, should do after trial rejection correct?
    %has to be all subjects for this script
    findFewestTrials (total_subs, conditions, acc)

    % IEM
    % need to check that the minimum number of trials is correct
    
    bands = [1:2];
    
    cd(analysis_dir)
    SpatialIEM_original_job(subjects)
    
    cd(analysis_dir)
    % Permuted IEM
    SpatialEM_Permute_job (subjects)
    
    
    cd(analysis_dir)
    % Average Tuning Functions, need to use all subjects for this function
    for iBand = 1:length(bands)
        avg_tf (total_subs, iBand, acc)
    end
    
    cd(analysis_dir)
    % Calculate Slopes
    for iCon = conditions
        calculateSlopes_Single_Freqs (total_subs, iCon)
        
        %reset analysis dir since cleared
        analysis_dir =  '/home/garrett/WTF_Bike/Analysis_Scripts';
        Compile_Slopes_Single_Freqs (total_subs, iCon)
    end
    
    analysis_dir =  '/home/garrett/WTF_Bike/Analysis_Scripts';
    cd(analysis_dir)
    
    
    
    %only looking at accurate trials
    acc_trials = 1;
    if acc_trials == 1
        accuracyAnalysis_job(subjects,total_subs)
        acc = 1;
    end
    
    %change detection d'
    Compute_Dprime(total_subs)
    
    %IEM generalization
    TTall_job = SpatialEM_TTall_job(subjects,acc);
    TTall_permJob = SpatialEM_TTall_Permute_job(subjects,acc);
    wait(TTall_permJob,'finished');
    CrossTT_job = SpatialIEM_Cross_TT_job(subjects,acc);
    wait(CrossTT_job,'finished');
    CrossTT_permJob = SpatialIEM_Cross_TTpermute_job(subjects,acc);   
    wait(CrossTT_permJob,'finished');
    
    for iCon = conditions
        calculateSlopes_Single_Freqs_TTall(subjects, iCon, acc)   
    end
    
    calculateSlopes_Single_Freqs_TTcross (subjects,acc) 
    calculateSlopes_Single_Freqs_TTcross_perm (subjects,acc) % ^permute
    
    header = 'Analysis Complete!';
    message = 'No errors occured in the script! Praise God.';
    sendEmailToMe(header,message)
    
    %Continous WM Task
    % Model behavioral data
    %Beh_WM_Modelling_job(subjects)
    
    %analysis_dir =  '/home/garrett/WTF_Bike/Analysis_Scripts';
    %cd(analysis_dir)
    % Analyze & plot beh data, have to have total subjects here
    %Beh_WM_Model_Analysis (subjects,0)
    
    
catch e
    
    header = 'Error in Analysis Script. FUCK.';
    message = sprintf('The identifier was:\n%s.\nThe message was:\n%s', e.identifier, e.message);
    sendEmailToMe(header,message)
    error(message)
    
end
%% Plotting
cd(analysis_dir)
%indiv = 1; % 1 to plot individual surface plots, 
over = 1; % 1 to plot overall averaged plots

if plotting == 1 
    
    %control output of overall or individual plots
    individual = 0;
    
    for acc = 1
        % overall surface plots 
        [h,p,ci,stats]=subplot_script (subjects, individual, over, acc);

        %cd(analysis_dir)
        % 2D overall slope plots, first number is the band (1 = ALPHA, 2=THETA)
        Slope_Plots([1:2],[1:2],1,1,0,total_subs, acc,post_ex,low_first,rest_first)
        
        if acc == 1
            Compile_Slopes_Single_Freqs_TTall (total_subs, acc) % compile the slopes for ALL (redundant???)
            Compile_Slopes_AND_plot_Single_Freqs_TTcross (total_subs, acc)  % compile ^ and produce plots
            IEM_Gen_BF_RealvPerm_Plot
        end
    end
    
    
end

%% FFT and Spectographs
fft = 1;
if fft == 1
    fft_job = FFT_Script_Job (subjects);
    Spectra_Plot (total_subs, 0) %total_subs(setdiff(1:end,3) remove sj3
end

%% HILBERT TOPOPLOTS
for freq = 1:2 %(1=THETA, 2=ALPHA)
    Hilberts(total_subs,0,0,freq,1) %second value controls compiling data, third checking topoplots, 5 controls plotting 3D heads
end

return
%% Train Model on Both sets of Data
trainBoth_realJob = SpatialIEM_TrainBoth_Balance_job(total_subs);
wait(trainBoth_realJob,'finished');
trainBoth_permJob = SpatialIEM_TrainBoth_Permute_job(total_subs);
wait(trainBoth_permJob,'finished');

s = parcluster();
trainBoth_slopesJob = createJob(s);

for iSub = 1:length(total_subs)
    sjNum = total_subs(iSub);
    
    createTask(trainBoth_slopesJob,@calculateSlopes_Single_Freqs_TrainBoth,0,{sjNum});
end
submit(trainBoth_slopesJob)
wait(trainBoth_slopesJob,'finished')

for iCon = conditions
    Compile_Slopes_Single_Freqs_TrainBoth(total_subs,iCon)
end
% 
Slope_Plots_TrainBoth([1],[1],1,0,0,total_subs,0,0,0,0)
   
% Train Classifier on for theta and alpha activity in frontal vs
% parietal-occipital electrodes?


%% Use Range of Frequency Bands for IEM
job = Spatial_IEM_All_singleFs_job(total_subs);
wait(job,'finished');

job = Spatial_IEM_All_singleFs_Permute_job(total_subs);
wait(job,'finished');

job = computeSlopes_All_singleFreqs_job(total_subs);

for iCon = 1:2
    compile_Slopes_All_singleFreqs(total_subs,iCon)
end

slopes_All_singleFreqs_plot


%% Non Parametric statistics 

%----------------------------- TrainBoth ----------------------------------
%MAKE SURE TO CHANGE PERMUTATIONS OF Spatial_IEM_Perm_accTrials TO 1000
Acc_IEM_slope_job = Acc_IEM_slope_FreqPerm_job(subjects);
wait(Acc_IEM_slope_job,'finished')

TrainBoth_Acc_IEM_slope_job = TrainBoth_Acc_IEM_slopeFreqPerm_job (subjects);
wait(TrainBoth_Acc_IEM_slope_job,'finished')

s = parcluster;
compile_AccIEM_FreqStats_job = createJob(s);

for iModel = 1:2 % 1 = Independent, 2 = Fixed
    for iBand = 1:2
        createTask(compile_AccIEM_FreqStats_job, @compile_Acc_IEM_slopeFreqStats, 0,...
            {subjects,iBand,iModel});
    end
end
submit(compile_AccIEM_FreqStats_job);
wait(compile_AccIEM_FreqStats_job,'finished')

%Stats
compute_slopeStats_job = createJob(s);
for iModel = 1:2
    for iBand = 1:2
        for iWave = 1:2
            createTask(compute_slopeStats_job,@slopes_nonParm_clustering_mcc,0,{iBand,iWave,iModel});
        end
    end
end
submit(compute_slopeStats_job);
wait(compute_slopeStats_job,'finished');

% Plotting
for iModel = 2
    for iBand = 1
        TrainBoth_CRF_Plots(iBand)
        
        for iWave = 2
            nonParam_slopePlots(iBand,iWave,iModel)
        end
    end
end

% Probabilities of BF clusters for Train Both Balanced model
slope_nonParam_clusterProbs

%------------------------- Generalizations -------------------------------
TTall_perm_nonParm_job = SpatialEM_TTall_Permute_nonParam_job(total_subs);
wait(TTall_perm_nonParm_job,'finished');

TTall_slopes_nonParam_job = calculateSlopes_SingleFreqs_TTall_nonParam_job(total_subs,1);
wait(TTall_slopes_nonParam_job,'finished');

TTall_slopes_compileNonParam = CompileSlopes_SingleFreqs_TTallnonParam_job(total_subs);
wait(TTall_slopes_compileNonParam,'finished')

CrossTT_perm_nonParm_job = SpatialIEM_Cross_TTpermute_nonParam_job(total_subs,1);
wait(CrossTT_perm_nonParm_job,'finished');

TTcross_slopes_nonParam_job = calculateSlopes_Single_Freqs_TTcross_perm_nonParam_job(total_subs);
wait(TTcross_slopes_nonParam_job,'finished');

TTcross_compile_slopes_nonParam_job = Compile_Slopes_Single_Freqs_TTcross_nonParam_job(total_subs);
wait(TTcross_compile_slopes_nonParam_job,'finished');

genStats_job = computeGeneralization_nonParm_stats_job(1); %Alpha Generalizations
wait(genStats_job,'finished')

% Train Both Generalizations
TB_realGen_job = SpatialIEM_TrainBoth_Generalize_job (total_subs);
wait(TB_realGen_job,'finished');
TB_permGen_job = SpatialIEM_TrainBoth_Generalize_Permute_job (total_subs);
wait(TB_permGen_job,'finished');
TB_calcGen_slopes_job = TrainBoth_calculateGeneralize_slopes_job(total_subs);
wait(TB_calcGen_slopes_job,'finished');

TrainBoth_compileGen_slopes (total_subs) %maybe send to other node when using larger iterations

TB_genStats_job = computeTrainBoth_genStats_job;

plot_generalizations_nonParam

% ------------------------- Single Frequencies ---------------------------
TrainBoth_SingleFreqs_job = SpatialIEM_TrainBoth_Balance_SingleFreq_job(total_subs);
wait(TrainBoth_SingleFreqs_job,'finished');

TrainBoth_SingleFreqs_nonParam_Perm_job = SpatialIEM_TrainBoth_Balance_singleFs_Permute_nonParm_job(total_subs);
wait(TrainBoth_SingleFreqs_nonParam_Perm_job,'finished');

TrainBoth_singleFreqs_calcSlopes_job = TrainBoth_computeSlopes_All_singleFreqs_nonParam_job(total_subs);
wait(TrainBoth_singleFreqs_calcSlopes_job,'finished');

TrainBoth_compile_Slopes_All_singleFreqs(total_subs);

TrainBoth_singleFreqs_statsJob = computeStats_TrainBoth_SingleFreqs_job;

TrainBoth_slopes_All_singleFreqs_plot

%% Permute Time domain
timePerm_job = SpatialIEM_ACC_TimePerm_job(subjects);
wait(timePerm_job,'finished');
timePerm_slopes_job = calc_timePerm_IEM_slopes_job (subjects);
wait(timePerm_slopes_job,'finished');
compile_timePerm_job = compile_timePerm_slopes_job (subjects);


%% Split locations

%First run SpatialIEM_TrainBoth_Balance
splitTB_job = splitLoc_calculateSlopes_job(subjects);
compile_splitLoc_slopes (subjects,band)
splitLoc_slopePlots(band)