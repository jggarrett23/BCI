function Analysis_Pipeline

total_subjects = [1:13,15:25,27:33,35];
preprocess_subjects = [33,35];

% Merge Behavior
for sjNum = preprocess_subjects
    Merge_Beh_Data(sjNum);
end

% Preprocess EEG data
%eeg_preprocess_job = ERP_Preprocess_job(preprocess_subjects);

%hilb_preproc_job = hilb_preprocess_2_job(preprocess_subjects);
ica_preprocess_job = EEG_ICA_Preprocess_job(preprocess_subjects);

% Extract ERP
aggressive_filt = 1; % if data filtered with high pass of 0.01 vs 1
for sjNum = preprocess_subjects
    
    %Extract_CDA(sjNum, aggressive_filt)
    Extract_CDA(sjNum);
end

% Extract Theta and Alpha Power
hilb_job = extract_hilberts_job(preprocess_subjects);

% Compile Data
compile_behavior(total_subjects);
compile_erps(total_subjects);
stats_job = compute_stats_job;
compile_spectra(total_subjects);

% Plotting
plot_beh
plot_cda
plot_theta_Pw
plot_latAlpha

% Extra
pca_job = trialwise_pca_job(preprocess_subjects);


end