% genaffine.m is a routine that constructs a generalized affine approximation 
% to the solution of the small open economy in Coeurdacier, Rey, 
% and Winant (2011); the methodology is described in detail in 
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
name_model = 'CRW';

%% variables

% Enter name of state variables
syms N r y ;
zt = [N; r; y] ;
% Enter name of jump variables
syms c xr wr ;
yt = [c; xr; wr];
% Enter name of exogenous shocks
syms eps_r eps_y
epst = [eps_r; eps_y] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step1 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameters

% Enter name of parameters
syms sigmar sigmay beta gamma theta rhor rhoy rr yy
MODEL.parameters.params = [ sigmar; sigmay; beta; gamma; theta; rhor; rhoy; rr; yy ] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step2 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% System

% EXPECTATIONAL EQUATIONS
% Enter row by row function f(y_t,z_t,y_{t+1},z_{t+1})=hh(y_t,z_t)+f_3y_{t+1}+f_4z_{t+1} in FOCs: ln(E_texp[f(y_t,z_t,y_{t+1},z_{t+1})])=0
% Enter h(y_t,z_t)
hh1 = log(beta)+gamma*c;
hh2 = -xr;
hh3 = exp(r)-wr;
% Enter f_3y_t+f_4z_t (note timing: t not t+1)
ff1 = -gamma*c+r;
ff2 = r;
ff3 = 0;

% LAW OF MOTION OF STATE VARIABLES
% Enter row by row law of motion of state variables z_{t+1}=gg(y_t,z_t)+lambda(z_t)(E_{t+1}-E_t)y_{t+1}+sigma(z_t)\epsilon_{t+1}
gg1 = exp(y)-exp(c+xr)+N*exp(xr);
gg2 = (1-rhor)*rr+rhor*r;
gg3 = (1-rhoy)*yy+rhoy*y;

% Fill in nonzero elements of lambda(z_t)
lambdaz(1,3) = N-exp(c);
% Fill in nonzero elements of sigma(z_t)
sigmaz(2,1) = sigmar;
sigmaz(3,2) = sigmay;

% Enter cumulant generating function of shocks
CCDF = @(uu) .5*diag(uu*uu.') ;
    
%% Choose Calibration

% Enter values of parameters
freq = 4;
sigmar = .025;
sigmay = .025;
beta = .96;
gamma = 2;
theta = 1;
rhor = .9;
rhoy = .9;
rr = .01996; % calibration as in Coeurdacier et al. (2011) to achieve A = 0 under their approximate solution for the risky steady state
%rr = log(1/beta-.014); % original calibration in Coeurdacier et al. (2011)
yy = log(theta);

% Optional: provide an initial guess for the deterministic steady state
SS_x = [0;rr+.5*sigmar^2;exp(rr);theta;rr;yy];

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
shocks = eye(Neps);
var_plot = {'c','r','y'}; var_sel = NaN(length(var_plot),1);
for i=1:length(var_plot)
    var_sel(i,1) = fct_varfind(MODEL,var_plot{i});
end
if sim_opt == 1
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
 