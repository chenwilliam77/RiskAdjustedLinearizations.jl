% genaffine.m is a routine that constructs a generalized affine approximation 
% to the solution of Fernandez-Villaverde, Guerron-Quintana, Rubio-Ramirez,
% and Uribe (2011); the methodology is described in detail in 
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
name_model = 'FVGQRRU';

%% variables

% Enter name of state variables
syms zt kt Dt im1t xxt epstbt epsrt sigmatbt sigmart real ; 
zt = [kt; Dt; im1t; xxt; epstbt; epsrt; sigmatbt; sigmart] ;
% Enter name of jump variables
syms yt ct wt qt lt bt it real ; 
yt = [ct ; wt ; qt ; lt ; bt ; it];
% Enter name of exogenous shocks
syms epsxx epstb epsr epssigmatb epssigmar real ;
epst = [epsxx ; epstb ; epsr ; epssigmatb ; epssigmar] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step1 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameters

% Enter name of parameters
syms alpha delta nu eta beta omega PhiD phi r rhotb rhor rhosigmatb sigmatb etatb rhosigmar sigmar etar rhoxx sigmaxx D kappa real
    
MODEL.parameters.params = [ alpha ; delta ; nu ; eta ; beta ; omega ; PhiD ; phi ; r ; rhotb ; rhor ; rhosigmatb ; sigmatb ;...
        etatb ; rhosigmar ; sigmar ; etar ; rhoxx ; sigmaxx ; D ; kappa ] ;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    run fct_setup_step2 ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% System

% EXPECTATIONAL EQUATIONS
% Enter row by row function f(y_t,z_t,y_{t+1},z_{t+1})=hh(y_t,z_t)+f_3y_{t+1}+f_4z_{t+1} in FOCs: ln(E_texp[f(y_t,z_t,y_{t+1},z_{t+1})])=0
% Enter h(y_t,z_t)
hh1 = log(beta) + r + epstbt + epsrt + nu*ct + log(1 + PhiD*(Dt - D)) ;
hh2 = log(beta) + nu*ct - qt ;
hh3 = log(alpha*exp( ((1-alpha)/(alpha + eta))*log((1-alpha)/(omega)) - (((1-alpha)*eta)/(alpha + eta))*kt + (((1+eta)*(1-alpha))/(alpha+eta))*xxt- (((1-alpha)*nu)/(alpha+eta))*ct) + (1-delta)*exp(qt)) - wt ;
hh4 = log(beta) + nu*ct - 2*it - log( 1 - exp(qt) + exp(qt)*phi*( exp(it - im1t) - 1)*(1/2)*(3*exp(it - im1t) - 1) + exp(bt)) ;
hh5 = log(1 - exp(qt)*phi*(exp(it - im1t) -1)) - lt ;
hh6 = log(beta) + nu*ct - 2*it - bt ;
% Enter f_3y_t+f_4z_t (note timing: t not t+1)
    f3 = sym(zeros(Ny,Ny)) ;
    f3(1,1) = -nu ;
    f3(2,1) = -nu ; f3(2,2) = 1 ;
    f3(4,1) = -nu ; f3(4,4) = 1 ; f3(4,6) = 2 ; 
    f3(6,1) = -nu ; f3(6,6) = 2 ;
    f4 = sym(zeros(Ny,Nz)) ;
    ff = f3*yt + f4*zt ;
ff1 = ff(1) ;
ff2 = ff(2) ;
ff3 = ff(3) ;
ff4 = ff(4) ;
ff5 = ff(5) ;
ff6 = ff(6) ;
    clear f3 f4 ff
    
% LAW OF MOTION OF STATE VARIABLES
% Enter row by row law of motion of state variables z_{t+1}=gg(y_t,z_t)+lambda(z_t)(E_{t+1}-E_t)y_{t+1}+sigma(z_t)\epsilon_{t+1}
    gg = sym([...
            log( (1-delta)*exp(kt) + (1 - (phi/2)*(exp(it - im1t) -1)^2)*exp(it) ) ; ...
            exp(r + epstbt + epsrt)*(exp(ct) + exp(it) + Dt - exp(((1-alpha)/(alpha + eta))*log((1-alpha)/omega) - nu*((1-alpha)/(alpha + eta))*ct + alpha*((1+eta)/(alpha+eta))*kt + (1-alpha)*((1+eta)/(alpha+eta))*xxt) + (PhiD/2)*(Dt-D)^2) ; ...
            it ; ...
            rhoxx*xxt ; ...
            rhotb*epstbt ; ...
            rhor*epsrt ; ...
            (1-rhosigmatb)*sigmatb + rhosigmatb*sigmatbt ; ...
            (1-rhosigmar)*sigmar + rhosigmar*sigmart ; ...
        ]) ;
gg1 = gg(1) ;
gg2 = gg(2) ;
gg3 = gg(3) ;
gg4 = gg(4) ;
gg5 = gg(5) ;
gg6 = gg(6) ;
gg7 = gg(7) ;
gg8 = gg(8) ;
    clear gg
    
% Fill in nonzero elements of sigma(z_t)
sigmaz = sym(zeros(Nz,Neps)) ;
sigmaz(4,1) = sigmaxx ;
sigmaz(5,2) = exp(sigmatbt) ;
sigmaz(6,3) = exp(sigmart) ;
sigmaz(7,4) = etatb ;
sigmaz(8,5) = etar ;
    
    cholSigma = sym(eye(Neps,Neps)) ; cholSigma(3,5) = kappa ; cholSigma(3,3) = sqrt(1-kappa^2) ;
    % check where shocks go
    MODEL.shocks.epsilon = cholSigma*epst ;
    sigmaz = sigmaz*cholSigma ;
    
% Enter cumulant generating function of shocks
CCDF = @(uu) .5*diag(uu*uu.') ;
    
%% Choose Calibration
freq = 12 ;
% Enter values of parameters
Country = 'Argentina' ;
Model = 1 ;
[ MODEL ] = fct_FVGQRRU_calibration( MODEL, Country, Model ) ;

% Optional: provide an initial guess for the deterministic steady state
%SS_x = [];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
%    disp('---------------------Parametrizing---------------------')
%    Npar = length(MODEL.parameters.params);
%    for i=1:Npar;
%        eval(['params(i,1) = ' char(MODEL.parameters.params(i)) ';'])
%    end
    
    MODEL.calibration.freq = freq ;
%    MODEL.calibration.params = params ;
    
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

disp('---------------------DSS---------------------')    
disp(['              ss c = ' num2str(MODEL.solution.dss.y(1))])
disp(['              ss i = ' num2str(MODEL.solution.dss.y(6))])
disp(['              ss b = ' num2str(MODEL.solution.dss.y(5))])
disp(['slope c on sigmart = ' num2str(MODEL.solution.dss.Psi(1,8))])
disp(['slope i on sigmart = ' num2str(MODEL.solution.dss.Psi(6,8))])
disp(['slope q on sigmart = ' num2str(MODEL.solution.dss.Psi(3,8))])
disp('---------------------RSS---------------------')    
disp(['              ss c = ' num2str(MODEL.solution.rss.y(1))])
disp(['              ss i = ' num2str(MODEL.solution.rss.y(6))])
disp(['              ss b = ' num2str(MODEL.solution.rss.y(5))])
disp(['slope c on sigmart = ' num2str(MODEL.solution.rss.Psi(1,8))])
disp(['slope i on sigmart = ' num2str(MODEL.solution.rss.Psi(6,8))])
disp(['slope q on sigmart = ' num2str(MODEL.solution.rss.Psi(3,8))])

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
shocks = eye(Neps) ; kappa = fct_parfind(MODEL, 'kappa') ; shocks(3,3) = 1/sqrt(1-kappa^2) ;
var_plot = {'ct'; 'it'; 'Dt'; 'epsrt'}; var_sel = NaN(length(var_plot),1);
for i=1:length(var_plot);
    var_sel(i,1) = fct_varfind(MODEL,var_plot{i});
end
if sim_opt == 1;
    z0_rss = z_rss ;% [quantile(real(MODEL.stochsim.z(:,1)),.01/freq)';z_rss(2:end)] ; % start from a once every 100 years bad realization of hats
else
    z0_rss = z_rss ;
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
    filename = [char(MODEL.name), '_', MODEL.calibration.Country, '_M', num2str(MODEL.calibration.MM), '_results.mat'] ;
    save(filename,'MODEL')
    delete('model_setup.mat')
	disp('... done!')
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
