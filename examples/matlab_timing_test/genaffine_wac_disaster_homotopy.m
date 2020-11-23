% genaffine.m is a routine that constructs a generalized affine approximation
% to the solution of Wachter (2013) in discrete time; the methodology is
% described in detail in Pierlauro Lopez, David Lopez-Salido, and Francisco
% Vazquez-Grande, 2016, "Entropy-Based Affine Approximations of Dynamic
% Equilibrium Models".
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
name_model = 'EZdisaster';

%% Special options

% disp('Select rho (inverse EIS):')
% rho_opt = input('');
rho_opt = 2;

sol_opt = 0;

%% variables

% Enter name of state variables
syms p epsc epsxi ;
zt = [p; epsc; epsxi] ;
% Enter name of jump variables
syms vc xc rf ;
yt = [vc; xc; rf];
% Enter name of exogenous shocks
syms eps_p eps_c eps_xi
epst = [eps_p; eps_c; eps_xi] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step1 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameters

% Enter name of parameters
syms mu sigmaa nu delta rhop pp phip rho gamma beta
MODEL.parameters.params = [ mu; sigmaa; nu; delta; rhop; pp; phip; rho; gamma; beta ] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step2 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% System

% EXPECTATIONAL EQUATIONS
% Enter row by row function f(y_t,z_t,y_{t+1},z_{t+1})=hh(y_t,z_t)+f_3y_{t+1}+f_4z_{t+1} in FOCs: ln(E_texp[f(y_t,z_t,y_{t+1},z_{t+1})])=0
% Enter h(y_t,z_t)
hh1 = log(beta)-gamma*mu+gamma*nu*p-(rho-gamma)*xc+rf;
if rho_opt == 1
    hh2 = beta*xc-vc;
else
	hh2 = log(1-beta+beta*exp((1-rho)*xc))-(1-rho)*vc;
end
hh3 = (1-gamma)*(mu-nu*p-xc);
% Enter f_3y_t+f_4z_t (note timing: t not t+1)
ff1 = -gamma*sigmaa*epsc+gamma*nu*epsxi+(rho-gamma)*vc;
ff2 = 0;
ff3 = (1-gamma)*(vc+sigmaa*epsc-nu*epsxi);

% LAW OF MOTION OF STATE VARIABLES
% Enter row by row law of motion of state variables z_{t+1}=gg(y_t,z_t)+lambda(z_t)(E_{t+1}-E_t)y_{t+1}+sigma(z_t)\epsilon_{t+1}
gg1 = (1-rhop)*pp+rhop*p;
gg2 = 0;
gg3 = 0;
% Fill in nonzero elements of sigma(z_t)
sigmaz(1,1) = sqrt(p)*phip*sigmaa;
sigmaz(2,2) = 1;
sigmaz(3,3) = 1;

% Enter cumulant generating function of shocks
CCDF = @(uu) .5*uu(:,1).^2 + .5*uu(:,2).^2 + (exp(uu(:,3)+uu(:,3).^2*delta^2/2)-1-uu(:,3))*p ;

%% Choose Calibration

% Enter values of parameters
freq = 4;
mu = 0.0252/freq;
sigmaa = 0.020/sqrt(freq);
nu = .3;
delta = 0;
rhop = .080^(1/freq);
pp = .0355/freq;
phip = .0114/freq/sigmaa/sqrt(pp);
rho = rho_opt;
gamma = 3.0;
beta = exp(-0.012/freq);

% Optional: provide an initial guess for the deterministic steady state
if rho_opt == 1
    vcSS = (mu-nu*pp)*beta/(1-beta);
    xcSS = vcSS/beta;
else
    xcSS = log((1-beta)/(exp((1-rho)*(nu*pp-mu))-beta))/(1-rho);
    vcSS = xcSS+nu*pp-mu;
end
SS_x = [ vcSS; xcSS; -log(beta)+gamma*(mu-nu*pp)-(rho-gamma)*(vcSS-xcSS); pp; 0; 0 ];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % tic;
    % disp('---------------------Parametrizing---------------------')
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

	% disp('... done!')
    % toc;

    % disp('---------------------Loading---------------------')
	% tic;
    run fct_setup_step3 ;
    % toc;

    % save model_setup MODEL
    % clear
    % load model_setup ;
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

    % disp('... done!')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Solution

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % disp('---------------------Solving---------------------')
    run fct_solution ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Simulation

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % tic;
    % disp('------------------------------------------')
    % disp('Simulate? (=0 no; =1 yes)')
    % sim_opt = input('');
	sim_opt = 0;
    if sim_opt == 1;
        disp('How many periods?')
        TT = input('');
        disp('---------------------Simulating---------------------')
        [ MODEL ] = fct_stochsimul( MODEL, TT, floor(TT/100) ) ;
    end
	% disp('... done!')
    % toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% IRF

% Enter shocks for IRFs
horizon = 20;
shocks = eye(Neps);
var_plot = {'rf','vc'}; var_sel = NaN(length(var_plot),1);
for i=1:length(var_plot);
    var_sel(i,1) = fct_varfind(MODEL,var_plot{i});
end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % tic;
    % disp('---------------------Ex post impulse responses---------------------')
	% x = MODEL.solution.rss.x ;
    % Psi = MODEL.solution.rss.Psi ;
    % [ MODEL ] = fct_irf( MODEL, x, Psi, shocks, var_sel, var_plot, freq*horizon, z_rss ) ;
	% disp('... done!')
    % toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
