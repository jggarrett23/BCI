function TrainBoth_calculateGeneralize_slopes (sjNum)
%==========================================================================
%{

%}
%==========================================================================

close all

% setup directories
root = '/home/garrett/WTF_Bike/'; %out = 'AnalysisScripts/trunk/MATLAB';
dRoot = [root,'Data/TrainBoth/'];

     

load_suffix = '_SpatialTF_ALPHA_Gen.mat'; % name of files to be saved
saveName = '_CTFslopes_ALPHA_Gen.mat';


% grab subject's d
fName = [dRoot,sprintf('sj%02d_TrainBoth_changeDect_acc_Permute',sjNum), load_suffix];
tmp = load(fName);
em = tmp.em;
tmp.em = [];

%Rest
rDat.rest.total = em.tfs.total.rest;

%Exercise
rDat.ex.total = em.tfs.total.ex;

pDat.rest.total = em.permtfs.total.rest;

pDat.ex.total = em.permtfs.total.ex;

% Specify properties
nIter = size(pDat.rest.total,1); 
nGenSamps = size(rDat.rest.total,2);


x = 0:45:180; % WTF use real angular values

rSl.rest.total = nan(nGenSamps,nGenSamps);
rSl.ex.total = rSl.rest.total;

pSl.rest.total = nan(nIter,nGenSamps,nGenSamps);
pSl.ex.total = pSl.rest.total;

% real total data
for trSamp = 1:nGenSamps
    for teSamp = 1:nGenSamps
        rest_dat = rDat.rest.total(trSamp,teSamp,:);
        ex_dat = rDat.ex.total(trSamp,teSamp,:);
        %dat = squeeze(rDat.evoked(f,samp,:));
        
        rest_d = [rest_dat(1),mean([rest_dat(2),rest_dat(8)]),mean([rest_dat(3),rest_dat(7)]),mean([rest_dat(4),rest_dat(6)]),rest_dat(5)];
        rest_fit = polyfit(x,rest_d,1);
        rSl.rest.total(trSamp,teSamp)= rest_fit(1);
        
        
        ex_d = [ex_dat(1),mean([ex_dat(2),ex_dat(8)]),mean([ex_dat(3),ex_dat(7)]),mean([ex_dat(4),ex_dat(6)]),ex_dat(5)];
        ex_fit = polyfit(x,ex_d,1);
        rSl.ex.total(trSamp,teSamp)= ex_fit(1);
    end
end

% permuted total
for iIter = 1:nIter
    for trSamp = 1:nGenSamps
        for teSamp = 1:nGenSamps
            rest_dat = squeeze(pDat.rest.total(iIter,trSamp,teSamp,:));
            ex_dat = squeeze(pDat.ex.total(iIter,trSamp,teSamp,:));
            
            rest_d = [rest_dat(1),mean([rest_dat(2),rest_dat(8)]),mean([rest_dat(3),rest_dat(7)]),mean([rest_dat(4),rest_dat(6)]),rest_dat(5)];
            rest_fit = polyfit(x,rest_d,1);
            pSl.rest.total(iIter,trSamp,teSamp)= rest_fit(1);
            
            
            ex_d = [ex_dat(1),mean([ex_dat(2),ex_dat(8)]),mean([ex_dat(3),ex_dat(7)]),mean([ex_dat(4),ex_dat(6)]),ex_dat(5)];
            ex_fit = polyfit(x,ex_d,1);
            pSl.ex.total(iIter,trSamp,teSamp)= ex_fit(1);
        end
    end
end

% save slope matrices
filename = [dRoot,sprintf('sj%02d_TrainBoth_changeDect',sjNum),saveName];
save(filename,'rSl','pSl','-v7.3');


end