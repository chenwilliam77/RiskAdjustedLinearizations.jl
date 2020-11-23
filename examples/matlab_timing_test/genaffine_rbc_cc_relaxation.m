% genaffine_cc.m is a routine that constructs a generalized affine approximation
% to the solution of Jermann (1998) with Campbell-Cochrane habits; the
% methodology is described in Pierlauro Lopez, David Lopez-Salido, and
% Francisco Vazquez-Grande, 2016, "Entropy-Based Affine Approximations of
% Dynamic Equilibrium Models".
% Please refer to this work if using this code and cite it accordingly.
%
% Calls: fct_setup_step1.m, fct_setup_step2.m, fct_setup_step3.m,
%        fct_myfun.m, fct_solution.m, qzdecomp.m, fct_stochsimul.m, fct_irf.m
% Optional calls: fct_parfind.m, fct_varfind.m, vec.m
%
% Written by Pierlauro Lopez and Francisco Vazquez-Grande.
% (c) This version: October 2017.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear
    close all
    clc
    addpath('genaffine_functions')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Enter name of model
name_model = 'RBCCC';

%% Special options

% disp('Select CRRA (=0) or CC habits (=1):')
% hab_opt = input(''); %=0 no habits; =1 habits
hab_opt = 1;
if hab_opt == 0
    name_model = [name_model, '_CRRA'] ;
elseif hab_opt == 1
    name_model = [name_model, '_CChabits'] ;
end
sol_opt = 1;

%% variables

% Enter name of state variables
syms ka hats ;
zt = [ka ; hats] ;
% Enter name of jump variables
disp('Select number of strips')
% N = input(''); % number of strips (if used)
N = 0;
name_model = [name_model, '_strips',num2str(N)] ;

syms ik ck log_D_plus_Q riskfree q log_E_RQ
yt = [ck; ik; log_D_plus_Q; riskfree; q; log_E_RQ] ;
if N > 0
    q_div = sym('q_div_%d', [1 N]).';
    syms div r_div wres
    yt = [yt; div; r_div; wres; q_div];
end
% Enter name of exogenous shocks
syms eps_a
epst = [eps_a] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step1 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameters

% Enter name of parameters
syms IKbar beta delta mu alpha xi3 sigmaa gamma rhos S real
MODEL.parameters.params = [ IKbar; beta; delta; mu; alpha; xi3; sigmaa; gamma; rhos; S] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step2 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% System

    % Personalized input: some auxiliary variables
    syms xxx real
    Phi = symfun(exp(mu)-1 + delta/(1-xi3^2) + (IKbar/(1-1/xi3))*((xxx/IKbar)^(1-1/xi3)),xxx) ;
    Phi_1prime = diff(Phi,xxx,1) ;
    mt   = -log(beta) -gamma*( ck + (hab_opt == 1)*hats - log(1 +  Phi(exp(ik)) - delta) ) ;
    mtp1 = -gamma*(ck + (hab_opt == 1)*hats) ;
    Y    = exp(ck) + exp(ik) ;
    DivQ = alpha*Y - exp(ik) + (Phi(exp(ik)) - delta)*exp(q) ;

% EXPECTATIONAL EQUATIONS
% Enter row by row function f(y_t,z_t,y_{t+1},z_{t+1})=hh(y_t,z_t)+f_3y_{t+1}+f_4z_{t+1} in FOCs: ln(E_texp[f(y_t,z_t,y_{t+1},z_{t+1})])=0
% Enter h(y_t,z_t)
hh1 = log(Y) - (alpha-1)*ka ;
hh2 = -log(Phi_1prime(exp(ik))) - q ;
hh3 = log(DivQ + exp(q)) - log_D_plus_Q ;
hh4 = -q - log_E_RQ ;
if N > 0
    hh5 = log(sum(exp(q_div(1:N))) + exp(r_div)) - q ;
    hh6 = log(DivQ) - div ;
    hh7 = log(exp(q_div(N)) + exp(r_div)) - wres ;
    hh8 = - mt - r_div ;
    hh9 = - mt - (-riskfree) ;
    stripstart = 10; stripclasses = 1; % indicate number of row at which strips start and how many classes of strips to include
    for i=1:N
        eval(['hh' num2str(stripstart-1+i)   ' = ' char(-mt) ' - q_div(' num2str(i) ');']) ;
    end
else
    hh5 = - mt - q ;
    hh6 = - mt - (-riskfree) ;
end

% Enter f_3y_t+f_4z_t (note timing: t not t+1)
ff1 = 0 ;
ff2 = 0 ;
ff3 = 0 ;
ff4 = log_D_plus_Q ;
if N > 0
    ff5 = 0 ;
    ff6 = 0 ;
    ff7 = 0 ;
    ff8 = mtp1 + wres ;
    ff9 = mtp1 ;
    eval(    ['ff' num2str(stripstart)     ' = ' char(mtp1) ' + div ;']) ;
    for i=2:N
        eval(['ff' num2str(stripstart+i-1) ' = ' char(mtp1) ' + q_div(' num2str(i-1) ');']) ;
    end
else
    ff5 = mtp1 + log_D_plus_Q ;
    ff6 = mtp1 ;
end

% LAW OF MOTION OF STATE VARIABLES
% Enter row by row law of motion of state variables z_{t+1}=gg(y_t,z_t)+lambda(z_t)(E_{t+1}-E_t)y_{t+1}+sigma(z_t)\epsilon_{t+1}
gg1 = -mu + ka + log((1-delta) + Phi(exp(ik))) ;
gg2 = rhos*hats ;

% Fill in nonzero elements of lambda(z_t)
lambdaz(2,1) = 1/S*sqrt(1-2*hats)-1 ;

% Fill in nonzero elements of sigma(z_t)
sigmaz(1,1) = -sigmaa ;
sigmaz(2,1) = 0 ;

% Enter cumulant generating function of shocks
CCDF = @(uu) .5*diag(uu*uu.') ;

%% Choose Calibration

% Enter values of parameters
freq = 4;
alpha = 0.36 ;  % Jerman 98
xi3 = 0.23 ;  % Jerman 98
delta = (.025*(1+1/xi3))*(4/freq) ;  % Jerman 98
IKbar = delta/(1+1/xi3);  % Jerman 98
mu = (4*log(1.005))/freq ; % Jerman 98
sigmaa = ((sqrt(4)*.0064)/sqrt(freq))/(1-alpha) ;  % Jerman 98
gamma = 2 ; % From CC
rhos = .875^(1/freq) ; % From CC
rf = 0.0094/freq ; % From CC
xi1 = 0; % From CC
beta = (.9890^(4/freq)) ; % From Jerman 98
S = .057 ; % from CC

% Optional: provide an initial guess for the deterministic steady state
ik_SS = log(IKbar) ;
hats_SS = 0 ;
q_SS = 0 ;
wk_SS = -(log(beta) - gamma*mu - q_SS) ;
dk_SS = log(exp(wk_SS) - exp(q_SS))  ;
ck_SS = log(exp(dk_SS) + (1-alpha)*exp(ik_SS) -( exp(mu) - 1)*exp(q_SS)) - log(alpha) ;
ka_SS = log(exp(ck_SS) + exp(ik_SS))/(alpha-1)  ;
SS_x = double([ck_SS; ik_SS; wk_SS; ka_SS; 0 ]) ;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    disp('---------------------Parametrizing---------------------')
    Npar = length(MODEL.parameters.params);
    for i=1:Npar;
        eval(['params(i,1) = ' char(MODEL.parameters.params(i)) ';'])
    end

    MODEL.calibration.freq = freq ;
    MODEL.calibration.params = params ;

    if exist('SS_x','var')==1
        if length(SS_x) == Nx;
            MODEL.calibration.DSS = SS_x;
        else
            clear SS_x
        end
    end

	disp('... done!')
    toc;

    disp('---------------------Loading---------------------')
	tic;
    run fct_setup_step3 ;
    toc;

    save model_setup MODEL
    clear
    load model_setup ;
	Npar = length(MODEL.parameters.params);
    for i=1:Npar;
        eval([char(MODEL.parameters.params(i)) '=' num2str(MODEL.calibration.params(i)) ';'])
    end
    freq = MODEL.calibration.freq;

    zt = MODEL.variables.z ;
    yt = MODEL.variables.y ;
    xt = MODEL.variables.x ;
    epst = MODEL.shocks.epsilon ;
    Ny = length(yt) ;
    Nz = length(zt) ;
    Nx = length(xt) ;
    Neps = length(epst) ;

    disp('... done!')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Solution

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('---------------------Solving---------------------')
    run fct_solution ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Simulation

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    disp('------------------------------------------')
    disp('Simulate? (=0 no; =1 yes)')
    % sim_opt = input('');
    sim_opt = 0;
    if sim_opt == 1
        disp('How many periods?')
        TT = input('');
        disp('---------------------Simulating---------------------')
        [ MODEL ] = fct_stochsimul( MODEL, TT, floor(TT/100) ) ;
    end
	disp('... done!')
    toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% IRF

% Enter shocks for IRFs
horizon = 60;
shocks = .25*[ones(Neps, 1) -ones(Neps, 1)];
shocks(:,:,2) = .25*[ones(Neps, 1) -ones(Neps, 1)] ;
shocks(:,:,3) = .25*[ones(Neps, 1) -ones(Neps, 1)] ;
shocks(:,:,4) = .25*[ones(Neps, 1) -ones(Neps, 1)] ;
var_plot = {'ck', 'ik', 'q', 'riskfree', 'log_E_RQ', 'ka', 'hats'}; var_sel = NaN(length(var_plot),1);
for i=1:length(var_plot)
    var_sel(i,1) = fct_varfind(MODEL,var_plot{i});
end
z_rss = MODEL.solution.rss.z;
if sim_opt == 1
    z0_rss = z_rss ;
else
    z0_rss = z_rss;
end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    disp('---------------------Ex post impulse responses---------------------')
	x = MODEL.solution.rss.x ;
    Psi = MODEL.solution.rss.Psi ;
    [ MODEL ] = fct_irf( MODEL, x, Psi, shocks, var_sel, var_plot, freq*horizon, z0_rss ) ;
	disp('... done!')
    toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% save

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	disp('---------------------Saving---------------------')
    filename = [char(MODEL.name) '_results.mat'] ;
    save(filename,'MODEL')
    delete('model_setup.mat')
	disp('... done!')
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
