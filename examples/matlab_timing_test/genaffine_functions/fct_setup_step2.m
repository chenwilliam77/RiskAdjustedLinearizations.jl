% fct_setup_step2.m is a routine that sets up the required matrices for 
% a generalized affine approximation to the solution of a DSGE. Step 2/3.
%
% Called by: genaffine.m
%
% Written by Pierlauro Lopez and Francisco Vazquez-Grande.
% (c) This version: October 2017.

    lambdaz = sym(zeros(Nz,Ny)) ;
    sigmaz = sym(zeros(Nz,Neps)) ;    
    x_tilde = [y_tilde ; z_tilde] ;
	Psi_tilde = sym('Psi_tilde',[Ny Nz]);
    MODEL.variables.z = zt ;
    MODEL.variables.y = yt ;
    MODEL.variables.x = [yt ; zt] ;
    MODEL.shocks.epsilon = epst ;
    MODEL.policy.Psi_tilde = Psi_tilde ;
    MODEL.policy.z_tilde = z_tilde ;
    MODEL.policy.y_tilde = y_tilde ;
    MODEL.policy.x_tilde = x_tilde ;
    if exist('N','var') == 1;
        MODEL.opt.Nstrips = N;
    end
    % other auxiliary variables (if lambdaz is a function of y)
    for i=1:length(zt)
        eval(['syms ' char(z_tilde(i,:)) '_tilde'])            
        eval(['z_tilde_aux(i,1) = ' char(z_tilde(i,:)) '_tilde;'])
    end    