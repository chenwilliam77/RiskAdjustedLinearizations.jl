% qzdecomp.m is a routine that solves the quadratic matrix equation for 
% a linear perturbation around the deterministic steady state of a DSGE.
%
% Called by: genaffine.m
%
% Written by Pierlauro Lopez and Francisco Vazquez-Grande.
% (c) This version: October 2017. 

function Psi = qzdecomp(f1,f2,f3,f4,g1,g2,~)
    Ny = size(f4,1);
    Nz = size(f4,2);

    AA = [f4 f3;eye(Nz) zeros(Nz,Ny)];
    BB = [-f2 -f1;g2 g1];
    
    [SS,TT,QQ,ZZ] = qz(AA,BB,'complex');
    [~,~,~,ZZ] = ordqz(SS,TT,QQ,ZZ,'udo');
    Z11 = ZZ(1:Nz,1:Nz);
%    Z12 = ZZ(1:Nz,Nz+1:end);
    Z21 = ZZ(Nz+1:end,1:Nz);
%    Z22 = ZZ(Nz+1:end,Nz+1:end);
    Psi = Z21/Z11; Psi = real(Psi);
    if sum(abs(eig(AA,BB)) > 1) ~= Nz
        warning('1st-order perturbation around deterministic steady state is not saddle-path stable')
    else
        if nargin < 7
            disp('BK conditions for a unique locally bounded deterministic steady-state perturbation are satisfied')
        end
    end
end