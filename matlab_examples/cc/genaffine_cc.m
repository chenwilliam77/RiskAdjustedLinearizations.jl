% genaffine_cc.m is a routine that constructs a generalized affine approximation 
% to the solution of Campbell and Cochrane (1999); the methodology is described in
% Pierlauro Lopez, David Lopez-Salido and Francisco Vazquez-Grande, 2016,
% "Entropy-Based Affine Approximations of Dynamic Equilibrium Models".
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
    addpath('../../genaffine_functions')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Enter name of model
name_model = 'CC';
    
%% variables

% Enter name of state variables
syms hats epsa ;
zt = [hats; epsa] ;
% Enter name of jump variables
syms rf pc
disp('Select number of strips')
N = input(''); % number of strips (if used)
if N > 0
    pd_c = sym('pd_c_%d', [1 N]).';
    syms rd_c wc
    yt = [rf; pc; rd_c; wc; pd_c];
else
    syms wc
    yt = [rf; pc; wc];
end
% Enter name of exogenous shocks
syms eps_a
epst = [eps_a] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step1 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameters

% Enter name of parameters
syms beta ga sigmaa gamma rhos S
MODEL.parameters.params = [ beta; ga; sigmaa; gamma; rhos; S ] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step2 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% System

% EXPECTATIONAL EQUATIONS
% Enter row by row function f(y_t,z_t,y_{t+1},z_{t+1})=hh(y_t,z_t)+f_3y_{t+1}+f_4z_{t+1} in FOCs: ln(E_texp[f(y_t,z_t,y_{t+1},z_{t+1})])=0
% Enter h(y_t,z_t)
hh1 = log(beta*exp(-gamma*ga)) + gamma*hats + rf ;
if N > 0
    hh2 = log(sum(exp(pd_c(1:N))) + exp(rd_c)) - pc ;
    hh3 = log(beta*exp((1-gamma)*ga)) + gamma*hats - rd_c;
    hh4 = log(exp(pd_c(N))+exp(rd_c)) - wc;
    stripstart = 5; stripclasses = 1; % indicate number of row at which strips start and how many classes of strips to include
    for i=1:N
        eval(['hh' num2str(4+i)   ' = log(beta*exp((1-gamma)*ga)) + gamma*hats - pd_c(' num2str(i) ');']) ;
    end
else
    hh2 = log(beta*exp((1-gamma)*ga)) + gamma*hats - pc ;
    hh3 = log(1+exp(pc))-wc;
end
% Enter f_3y_t+f_4z_t (note timing: t not t+1)
ff1 = -gamma*hats -gamma*sigmaa*epsa;
if N > 0
    ff2 = 0;
    ff3 = -gamma*hats + (1-gamma)*sigmaa*epsa + wc;
    ff4 = 0;
    eval(['ff' num2str(4+1)       ' = -gamma*hats + (1-gamma)*sigmaa*epsa;']) ;
    for i=2:N
        eval(['ff' num2str(4+i)   ' = -gamma*hats + (1-gamma)*sigmaa*epsa + pd_c(' num2str(i-1) ');']) ;
    end
else
    ff2 = -gamma*hats + (1-gamma)*sigmaa*epsa + wc ;
    ff3 = 0;
end

% LAW OF MOTION OF STATE VARIABLES
% Enter row by row law of motion of state variables z_{t+1}=gg(y_t,z_t)+lambda(z_t)(E_{t+1}-E_t)y_{t+1}+sigma(z_t)\epsilon_{t+1}
gg1 = rhos*hats ;
gg2 = 0 ;
% Fill in nonzero elements of sigma(z_t)
sigmaz(1,1) = (1/S*sqrt(1-2*hats)-1)*sigmaa ;
sigmaz(2,1) = 1 ;

% Enter cumulant generating function of shocks
CCDF = @(uu) .5*diag(uu*uu.') ;
    
%% Choose Calibration

% Enter values of parameters
freq = 4;
ga = .0189/freq;
sigmaa = .015/sqrt(freq);
gamma = 2;
rhos = .875^(1/freq);
beta = exp(-.0094/freq+gamma*ga-0.5*gamma*(1-rhos));
xi = 0;
S = sqrt(gamma*sigmaa^2/(1-rhos-xi/gamma));

% Optional: provide an initial guess for the deterministic steady state
SS_x = [-log(beta*exp(-gamma*ga));log(beta*exp((1-gamma)*ga)/(1-beta*exp((1-gamma)*ga)));[1:N]'*log(beta*exp((1-gamma)*ga));[1:N]'*log(beta*exp((1-gamma)*ga))+log(beta*exp((1-gamma)*ga)/(1-beta*exp((1-gamma)*ga)));0;0];

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

%% Personalize output    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('---------------------Some results---------------------')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try
	Ns = MODEL.opt.Nstrips;
catch
    Ns = 0;
end    
disp(['Number of strips = ' num2str(Ns)])
disp('---------------------DSS---------------------')    
disp(['mean risk-free rate               = ' num2str(MODEL.solution.dss.y(1))])
disp(['mean log price-consumption ratio  = ' num2str(MODEL.solution.dss.y(2))])
disp(['slope risk-free rate              = ' num2str(MODEL.solution.dss.Psi(1,1))])
disp(['slope log price-consumption ratio = ' num2str(MODEL.solution.dss.Psi(2,1))])
disp('---------------------RSS---------------------')    
disp(['mean risk-free rate               = ' num2str(MODEL.solution.rss.y(1))])
disp(['mean log price-consumption ratio  = ' num2str(MODEL.solution.rss.y(2))])
disp(['slope risk-free rate              = ' num2str(MODEL.solution.rss.Psi(1,1))])
disp(['slope log price-consumption ratio = ' num2str(MODEL.solution.rss.Psi(2,1))])

%% Simulation

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    disp('------------------------------------------')
    disp('Simulate? (=0 no; =1 yes)')
    sim_opt = input('');
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
horizon = 10; % in years
shocks = [ones(Neps, 1) -ones(Neps, 1)];
var_plot = {'pc'}; var_sel = NaN(length(var_plot),1);
for i=1:length(var_plot)
    var_sel(i,1) = fct_varfind(MODEL,var_plot{i});
end
z_rss = MODEL.solution.rss.z; % starting point for IRFs

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    disp('---------------------Ex post impulse responses---------------------')
	x = MODEL.solution.rss.x ;
    Psi = MODEL.solution.rss.Psi ;
    [ MODEL ] = fct_irf( MODEL, x, Psi, shocks, var_sel, var_plot, freq*horizon, z_rss ) ;
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
    