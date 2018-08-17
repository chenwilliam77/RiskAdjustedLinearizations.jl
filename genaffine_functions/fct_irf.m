% fct_irf.m is a routine that constructs generalized impulse response functions
% under a generalized affine approximation to the solution of a DSGE.
%
% Called by: genaffine.m
%
% Written by Pierlauro Lopez and Francisco Vazquez-Grande.
% (c) This version: October 2017.

function [ MODEL ] = fct_irf( MODEL, x_rss, Psi_rss, shocks, var_sel, var_plot, TT, z_start ) 
    
%% Set-up
    
    params = MODEL.parameters.params ;
    params_val = MODEL.calibration.params ;
    
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
    Nfig = size(shocks,2) ;
    Ntau = size(shocks,3) ;
    
    zloading_fun_rss = matlabFunction(subs(MODEL.nonlinear.z_loading_on_epsilon,MODEL.policy.Psi_tilde(:),Psi_rss(:)),'vars',{xt}) ;    
    
    y_rss = x_rss(1:Ny) ;
    z_rss = x_rss((Ny+1):Nx) ;
     
    % from system (4) in the paper our system is
    % z(t+1) = z_tilde + g1_tilde*(y(t)-y_tilde) + g2_tilde*(z(t)-z_tilde) + ((I - lambda(z(t))Psi)^{-1})*sigma(z(t))*eps(t+1) 
    ztp1_rss = @(yt,zt,epsilon) z_rss + g1_rss*(yt-y_rss) + g2_rss*(zt-z_rss) + zloading_fun_rss([yt;zt])*epsilon ;
    
%% IRFs 
    IRF_rss = NaN(TT+1,Nx,Nfig) ;
    dIRF_rss = NaN(TT,Nx,Nfig) ;
    
    if nargin < 8;
        z_start = z_rss ;
    end
    
    for i=1:Nfig
        % t = 0
        z0_rss = z_start ;
        y0_rss = y_rss + Psi_rss*(z0_rss - z_rss) ;
        IRF_rss(1,:,i) = [y0_rss ; z0_rss] ;
        % Shocked periods
        for t=2:(Ntau)
            eps0 = shocks(:,i,t-1) ;
            z1_rss = ztp1_rss(y0_rss,z0_rss,eps0) ;
            y1_rss = y_rss + Psi_rss*(z1_rss - z_rss)  ;
            IRF_rss(t,:,i) = [y1_rss ; z1_rss] ;
            dIRF_rss(t-1,:,i) = IRF_rss(t,:,i) - IRF_rss(t-1,:,i) ;
            z0_rss = z1_rss ;
            y0_rss = y1_rss ;
        end
        % Last Shocked period and unshocked periods
        eps0 = shocks(:,i,Ntau) ;
        for t=(Ntau+1):(TT+1)
            z1_rss = ztp1_rss(y0_rss,z0_rss,eps0) ;
            y1_rss = y_rss + Psi_rss*(z1_rss - z_rss)  ;
            IRF_rss(t,:,i) = [y1_rss ; z1_rss] ;
            dIRF_rss(t-1,:,i) = IRF_rss(t,:,i) - IRF_rss(1,:,i) ;
            z0_rss = z1_rss ;
            y0_rss = y1_rss ;
            
            eps0 = zeros(Neps,1) ;
        end
	end


%% Plots
    freq = MODEL.calibration.freq ;
    for j = 1:Nfig
        figure;
        count = 0 ;
        for i = var_sel'
            count = count + 1 ;
            subplot(ceil(length(var_plot)/3),3,count)
            plot([1:TT]/freq,dIRF_rss(:,i,j),'b','linewidth',2)
            axis([1/freq TT/freq min([-1d-3;dIRF_rss(:,i,j)]) max([1d-3;dIRF_rss(:,i,j)])])
            title(var_plot{count})
        end
        eval(['print -deps2c pics/figirf_shock' num2str(j) '.eps'])
        eval(['print -dpdf pics/figirf_shock' num2str(j) '.pdf'])
    end

    
%% Save
    MODEL.irf = dIRF_rss ;
    
end
    
    