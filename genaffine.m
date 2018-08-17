% genaffine.m is a routine that constructs a generalized affine approximation 
% to the solution of a DSGE; the methodology is described in detail in 
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
    addpath('genaffine_functions')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Enter name of model
name_model = 'CC';

%% variables

% Enter name of state variables
syms z_name1 z_name2 ;
zt = [z_name1; z_name2] ;
% Enter name of jump variables
syms y_name1 y_name2 ;
disp('Select number of strips')
N = input(''); % number of strips (if used)
pd_d = sym('pd_d_%d', [1 N]).';
rd_d = sym('rd_d_%d', [1 N]).';
yt = [y_name1; y_name2 ; pd_d; rd_d];
% Enter name of exogenous shocks
syms eps_1
epst = [eps_1] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step1 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameters

% Enter name of parameters
syms par1 par2
MODEL.parameters.params = [ par1; par2 ] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step2 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% System

% EXPECTATIONAL EQUATIONS
% Enter row by row function f(y_t,z_t,y_{t+1},z_{t+1})=hh(y_t,z_t)+f_3y_{t+1}+f_4z_{t+1} in FOCs: ln(E_texp[f(y_t,z_t,y_{t+1},z_{t+1})])=0
% Enter h(y_t,z_t)
hh1 = ;
hh2 = ;
% if strips are included, always conclude with equations characterizing strips
for i=1:N
    stripstart = ; stripclasses = ; % if strips are included, indicate here number of row at which strips start and how many classes of strips to include
	eval(['hh' num2str(4+i) ' = ... - pd_a(' num2str(i) ');']) ;
	eval(['hh' num2str(4+N+i) ' = ... - rd_a(' num2str(i) ');']) ;    
end
% Enter f_3y_t+f_4z_t (note timing: t not t+1)
ff1 = ;
ff2 = ;
eval(['ff' num2str(4+1) ' = ...;']) ;
eval(['ff' num2str(4+N+1) ' = ... + vau;']) ;
for i=2:N
	eval(['ff' num2str(4+i) ' =... + pd_a(' num2str(i-1) ');']) ;
	eval(['ff' num2str(4+N+i) ' = ... + rd_a(' num2str(i-1) ');']) ;
end

% LAW OF MOTION OF STATE VARIABLES
% Enter row by row law of motion of state variables z_{t+1}=gg(y_t,z_t)+lambda(z_t)(E_{t+1}-E_t)y_{t+1}+sigma(z_t)\epsilon_{t+1}
gg1 = ;
gg2 = ;
% Fill in nonzero elements of lambda(z_t)
lambdaz(1,1) = ;
% Fill in nonzero elements of sigma(z_t)
sigmaz(1,1) = ;

% Enter cumulant generating function of shocks
CCDF = @(uu) .5*diag(uu*uu.') ;
    
%% Choose Calibration

% Enter values of parameters
freq = 4;
par1 = ;
par2 = ;

% Optional: provide an initial guess for the deterministic steady state
SS_x = [   ];

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
    sim_opt = input('');
    if sim_opt == 1;
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
var_plot = {'z_name1','y_name1'}; var_sel = NaN(length(var_plot),1);
for i=1:length(var_plot);
    var_sel(i,1) = fct_varfind(MODEL,var_plot{i});
end
if sim_opt == 1;
    z0_rss = [quantile(real(MODEL.stochsim.z(:,1)),.01/freq)';z_rss(2:end)] ; % start from a once every 100 years bad realization of hats
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
 