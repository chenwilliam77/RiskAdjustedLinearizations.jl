% fct_stochsimul.m is a routine that simulates the approximate solution 
% under a generalized affine approximation to the solution of a DSGE.
%
% Called by: genaffine.m
%
% Written by Pierlauro Lopez and Francisco Vazquez-Grande.
% (c) This version: October 2017. 

function [ MODEL ] = fct_stochsimul( MODEL, T, burnin )

    x_rss   = MODEL.solution.rss.x ;
    y_rss   = MODEL.solution.rss.y ;
    z_rss   = MODEL.solution.rss.z ;
    Psi_rss = MODEL.solution.rss.Psi ;
     
    params = MODEL.parameters.params ;
    params_val = MODEL.calibration.params ;
    TT = T + burnin ;
    
    g1_rss = MODEL.solution.rss.g1 ;
    g2_rss = MODEL.solution.rss.g2 ;

    zt = MODEL.variables.z ;
    yt = MODEL.variables.y ;
    xt = MODEL.variables.x ;
    epst = MODEL.shocks.epsilon ;
    Ny = length(yt) ;
    Nz = length(zt) ;
    Nx = length(xt) ;
    Neps = length(epst) ;
    
    zloading_fun_rss = MODEL.solution.rss.zloading_fun ;
     
    % from system (4) in the paper our system is
    % z(t+1) = z_tilde + g1_tilde*(y(t)-y_tilde) + g2_tilde*(z(t)-z_tilde) + ((I - lambda(z(t))Psi)^{-1})*sigma(z(t))*eps(t+1) 
    ztp1_rss = @(yt,zt,epsilon) z_rss + g1_rss*(yt-y_rss) + g2_rss*(zt-z_rss) + zloading_fun_rss([yt;zt])*epsilon ; 
    % check


%% simulate
    sims = NaN(T,Nx) ;
    z0 = z_rss ;
    y0 = y_rss ;
    randn('seed',1);
    for t=1:TT
        epsilon = randn(Neps,1) ;
        z1 = ztp1_rss(y0,z0,epsilon) ;
        y1 = y_rss + Psi_rss*(z1 - z_rss)  ;
        if t > burnin
            sims((t-burnin),:) = [y1 ; z1]' ;
        end
        z0 = z1 ;
        y0 = y1 ;
    end

%% save
    MODEL.stochsim.x = sims ;
    MODEL.stochsim.y = sims(:,(1:Ny)) ;
    MODEL.stochsim.z = sims(:,(1+Ny):Nx) ;
    
    MODEL.stochsim.x_sss = mean(sims)' ;
    MODEL.stochsim.y_sss = mean(sims(:,(1:Ny)))' ;
    MODEL.stochsim.z_sss = mean(sims(:,(1+Ny):Nx))' ;
    
end
    
    