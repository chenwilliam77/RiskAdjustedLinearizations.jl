% fct_setup_step1.m is a routine that sets up the required matrices for 
% a generalized affine approximation to the solution of a DSGE. Step 1/3.
%
% Called by: genaffine.m
%
% Written by Pierlauro Lopez and Francisco Vazquez-Grande.
% (c) This version: October 2017.

    MODEL.name = name_model;
	xt = [yt; zt] ;
    Nz = length(zt) ;
    Ny = length(yt) ;
    Nx = Nz + Ny ;
    Neps = length(epst) ;
    err_aux = length(union(union(intersect(zt,yt),intersect(zt,epst)),intersect(yt,epst)));
    if err_aux > 0
        error('Repeated variable names: change name of variables in vector zt, yt or epst to ensure no multiplicity.')
    end
    
    % Solution we are looking for has shape y_t = y_tilde + Psi_tilde(z_t-z_tilde)
    z_tilde = zt;
	y_tilde = yt;