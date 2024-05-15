function [Phi, omega, lambda, b, X_hat, S, mode_hz, Atilde, Btilde] = DMD(X,time,dt,varargin)

p = inputParser;
validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
validScalarPosInt = @(x) isnumeric(x) && isscalar(x) && (x > 0) && floor(x)==x;
validScalarInt = @(x) isnumeric(x) && isscalar(x) && floor(x) == x;

addRequired(p, 'X');
addRequired(p, 'time');
addRequired(p, 'dt');
addOptional(p, 'rbf_lift', 0);
addOptional(p, 'doDelay', 0);
addOptional(p, 'delayEmbedding', 0, validScalarPosInt);
addOptional(p, 'threshold', 0);
addOptional(p, 'scaling_modes', 0);
addOptional(p, 'verbose', 0);
addOptional(p, 'control',0);
addOptional(p, 'Y',[]);

parse(p, X, time, dt, varargin{:});

opts = p.Results;

X = opts.X;
time = opts.time;
dt = opts.dt;
doDelay = opts.doDelay;
h = opts.delayEmbedding;
rbf_lift = opts.rbf_lift;
threshold = opts.threshold;
scaling_modes = opts.scaling_modes;
verbose = opts.verbose;
control = opts.control;

time = time/1000;
X = double(X);

m = size(X,2)-1;
n = size(X,1);

% control input matrix
if control
   Y = opts.Y;
   
   if isempty(Y)
       error('No control matrix provided')
   end
end


% Lift Data

if doDelay
    % Augment data matrix to make n >> m (necessary for SVD)
    % by augmenting number of channels to hn with time shifted versions of
    % themselves
    if ~h
        h = ceil(((2*m)+1) / n);
        if h >= m
            h = 11;
        end
    end
    
    X_aug = [];
    
    if control
        Y_aug = [];
    end
    for i = 1:h
        X_aug = [X_aug;  X(:, i:end-h+i)];
        
        if control
            Y_aug = [Y_aug; Y(:, i:end-h+i)];
        end
    end
    X = X_aug;
    
    if control
        Y = Y_aug;
    end

end

if rbf_lift
   
    centers = -10:1:10;
    nCenters = length(centers);
    scale = 1; %scale parameter for radial basis function
    rbf_kernel = @(x,y,s) exp(-(norm(x-y).^2)/(2*s^2)); % radial basis function kernel
    
    X_temp = X; % first n rows are original data for recovery
    for iCenter = 1:nCenters
        temp = zeros(size(X));
        for i = 1:size(X,1)
            for j = size(X,2)
                temp(i,j) = feval(rbf_kernel, X(i, j), centers(iCenter), scale);
            end
        end
        
        X_temp = cat(1, X_temp, temp);
    end
    X = X_temp;
end

X1 = X(:,1:end-1);
X2 = X(:,2:end);

if control
    Y1 = Y(:, 1:end-1);
    Y2 = Y(:, 2:end);
    
    % concatenate state matrix with control input matrix
    X1_temp = X1; 
    X1 = [X1; Y1];
end


% U vectors are POD modes, columns are orthonormal
% S (i.e., sigma) is singular values indicating "importance" of each mode
[U, S, V] = svd(X1, 'econ');

% if a singular value is too small, the associated singular vectors u and v
% are so noisy that the component should not be included
beta_ratio = size(X1,1) / size(X1,2);

if beta_ratio > 1
    beta_ratio = 1/beta_ratio;
end

thresh = optimal_SVHT_coef(beta_ratio, 0) * median(diag(S));

if threshold
    r = sum(diag(S) > thresh);
else 
    r = size(U,2);
end

% % rank reduction
if r < size(U,2)
    Ur = U(:, 1:r);
    Sr = S(1:r, 1:r);
    Vr = V(:, 1:r);
else
    Ur = U;
    Sr = S;
    Vr = V;
end


% A is a matrix describing the time-dynamics of a continous-time sytem in a
% discrete time system. A = exp(AA*delta_t).
% A often intractable, but can evolve its dynamics in a low-rank subspace.
% Atilde is r x r projection matrix of A onto POD modes displaying low-rank dynamics
% Atilde defines a low-dimensional linear model of the dynamical system on
% POD coordinates
if control
    Ur_1 = Ur(1:size(X1_temp,1),:);
    Ur_2 = Ur(size(X1_temp,1)+1:end,:);
    
    
    [U_hat, S_hat, V_hat] = svd(X2, 'econ'); % SVD of time shifted matrix
    beta_ratio = size(X2,1) / size(X2,2);
    if beta_ratio > 1
        beta_ratio = 1/beta_ratio;
    end
    thresh = optimal_SVHT_coef(beta_ratio, 0) * median(diag(S)); % compute second truncation threshold
    r = sum(diag(S_hat) > thresh);
    
    r = size(U_hat,2);
    U_hat = U_hat(:,1:r);
    
    Atilde = U_hat'*X2*Vr/Sr*Ur_1'*U_hat;
    Btilde = U_hat'*X2*Vr/Sr*Ur_2';
else
    Atilde = Ur'*X2*Vr/Sr;
    Btilde = [];
end


% eigen decomposition of Atilde
% colums of W are eigenvectors, diag of Lambda are corresponding eigenvalues
if scaling_modes
    
    % scale modes by the magnitude of singular values
    Ahat = (Sr^(-1/2)) * Atilde * (Sr^(1/2));
    [W_hat, D] = eig(Ahat);
    W = Sr^(1/2) * W_hat;
else 
    [W, D] = eig(Atilde);
end

lambda = diag(D);

% magnitude and phase of lambda contain info about the time dynamics 
% rate of growth/decay and frequency of oscillation reflected in magnitude
% and phase of lambda, respectively. Sign of the real component determines
% if growing, decaying, or stable

% transform eigenvalues for convience
omega = log(lambda)/dt;

% reconstruct eigendecomposition of A from eigenvectors(W) and
% eigenvalues(D)
%Phi = X2*Vr/Sr*W; % DMD modes are columns (i.e. eigen vectors of A)
if control
    Phi = X2*Vr/Sr*Ur_1'*U_hat*W;
else
    Phi = X2*Vr/Sr*W;
end


% magnitude (i.e. abs) of Phi represents spatially coherent activation
% phase of Phi represents the the phase of this activation across
% electrodes
% mode amplitude of Phi is defined as the square of its vector magnitude
% (i.e., 2norm). P = diag(Phi'*Phi)

if control
    b = Phi\X1_temp(:,1);
else
    b = Phi\X1(:,1); % coefficients of initial condition x1 in eigenvector basis
end


% Projected future solution for all times in the future
Z = zeros(length(b), length(time));
for k=1:length(time)
    Z(:,k) = b .* lambda.^k;
end

% keep first n rows for analysis
Phi = Phi(1:n,:);

% convert phase of eigenvalues more interpretable and in units cycle/sec
% i.e. Hz
mode_hz = abs(imag(omega) / (2*pi));

% remove duplicate modes
[mode_hz, IA] = unique(mode_hz, 'stable');

Phi = Phi(:, IA);
omega = omega(IA);
lambda = lambda(IA);
b = b(IA);
Z = Z(IA,:);

keep_modesIdx = mode_hz <= 30;
    
Phi = Phi(:,keep_modesIdx);
omega = omega(keep_modesIdx);
b = b(keep_modesIdx);
lambda = lambda(keep_modesIdx);
mode_hz = mode_hz(keep_modesIdx);
Z = Z(keep_modesIdx,:);


time = time*1000;

X_hat = Phi*Z;

if verbose
    
    diag_fig = figure('units', 'inches');
    
    % plot singular values
    subplot(2,2,1)
    plot(diag(S)/sum(diag(S)), 'or', 'HandleVisibility', 'off');
    box off
    title('SVD Singular Values');
    if r
        xline(r+.5, '--');
        legend('Truncation Threshold');
        legend box off
    end
    
    % plot eigen values
    subplot(2,2,2)
    theta = (0:1:100)*2*pi/100;
    plot(cos(theta),sin(theta),'k--'); %unit circle
    hold on, grid on
    scatter(real(lambda), imag(lambda), 'ok');
    axis([-1.1, 1.1, -1.1, 1.1]);
    title('Eigenvalues');
    
    % plot DMD reconstruction vs True
    
    if n > 1
        
        tick_labels = linspace(min(time),max(time),5);
        
        subplot(2,2,3)
        imagesc(X);
        xticks(dsearchn(time',tick_labels'));
        xticklabels(tick_labels);
        title('True')
        
        subplot(2,2,4);
        imagesc(real(X_hat));
        xticks(dsearchn(time',tick_labels'));
        xticklabels(tick_labels);
        title('Reconstructed');
        
    else
        subplot(2,2,[3,4]);
        plot(time, X, 'Linewidth', 1.5); hold on
        plot(time, real(Z), '--r', 'Linewidth', 1.5);
        xlabel('Time (ms)')
        legend('True', 'Reconstructed');
        box off
        legend box off
        title({'DMD Reconstruction vs Ground Truth'; sprintf('Delay Steps = %d', h)});
    end
    
    diag_fig.Position = [2.1583, 2.4417, 8.7833, 4.9];
    
end

end
