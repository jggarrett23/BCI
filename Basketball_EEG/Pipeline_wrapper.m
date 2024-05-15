function Pipeline_wrapper

% select subjects
subjects = [11:14,16:39,51:56];

ICAcleaning_job = remove_ica_artifacts_job(subjects);


end