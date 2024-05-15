function job = Perm_hilb_classify_trial_job (subjects)

%% Settings
s = parcluster();
s.ResourceTemplate = '--ntasks=^N^ --cpus-per-task=^T^ --nodes=^N^ --job-name=pHILB_ML --exclusive';

job = createJob(s);

%% Run Job

for iSub = 1:length(subjects)
    
    sjNum = subjects(iSub);
    
       
    %Hilb_classify_trialType(sjNum)
    createTask(job, @Perm_hilb_classify_trialType, 0, {sjNum},...
        'Name', sprintf('s%d_%d_pHILB',sjNum));
        
    
end

submit(job)