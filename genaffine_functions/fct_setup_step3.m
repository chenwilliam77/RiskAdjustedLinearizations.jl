% fct_setup_step3.m is a routine that sets up the required matrices for 
% a generalized affine approximation to the solution of a DSGE. Step 3/3.
%
% Called by: genaffine.m
%
% Written by Pierlauro Lopez and Francisco Vazquez-Grande.
% (c) This version: October 2017.

	params = MODEL.parameters.params ;
    params_val = MODEL.calibration.params ;
	tx = MODEL.variables.x ;
	Psi = MODEL.policy.Psi_tilde;
	tPsi = Psi(:) ;
	tildevar = [tx;tPsi] ;
    
    CCDF = @(uu) subs(CCDF(uu),params,params_val) ;  

    try
        Ns = MODEL.opt.Nstrips;
    end

	disp('... building matrices...')
    hh = []; ff = [];
	for i=1:Ny
        eval(['hh = [hh; hh' num2str(i) '];']) ;
        eval(['ff = [ff; ff' num2str(i) '];']) ;
    end
    gg = [];
	for i=1:Nz
        eval(['gg = [gg; gg' num2str(i) '];']) ;
    end
    MODEL.nonlinear.h = hh ;
    MODEL.nonlinear.f = ff ;
    MODEL.nonlinear.g = gg ;
    
    if exist('stripstart','var') == 1
        MODEL.opt.stripstart = stripstart;
        MODEL.opt.stripclasses = stripclasses;
        h_aux = hh(1:stripstart-1);
        f_aux = ff(1:stripstart-1);
        for i = 1:stripclasses
            h_aux = [h_aux;hh(stripstart+(i-1)*Ns:stripstart+(i-1)*Ns+(Ns>1))];
            f_aux = [f_aux;ff(stripstart+(i-1)*Ns:stripstart+(i-1)*Ns+(Ns>1))];
        end
    else
        h_aux = hh;
        f_aux = ff;
    end
    
    hh = sym(subs(hh,params,params_val));
    h_aux = sym(subs(h_aux,params,params_val));
    f_aux = sym(subs(f_aux,params,params_val));
    gg = sym(subs(gg,params,params_val));
    
    lin_intercept = hh;
    lin_coefs_h = jacobian(h_aux,xt.');
	lin_coefs_f = jacobian(f_aux,xt.');
    f0 = fct_myfun(lin_intercept,{tx});
    if exist('stripstart','var') == 1
        [f1,f2] = fct_myfun(lin_coefs_h,{tx},[Ns;stripstart;stripclasses;-1]);
        [f3,f4] = fct_myfun(lin_coefs_f,{tx},[Ns;stripstart;stripclasses;1]);    
    else
        [f1,f2] = fct_myfun(lin_coefs_h,{tx},[Ny;Nz]);
        [f3,f4] = fct_myfun(lin_coefs_f,{tx},[Ny;Nz]);        
    end
    
    if norm(f3(zeros(Nx,1))-f3(ones(Nx,1)),Inf)+norm(f4(zeros(Nx,1))-f4(ones(Nx,1)),Inf) > 0
        error('Equations do not fit the generalized affine setup: f3 and f4 must be a function only of the deep parameters. Rewrite equations hh and ff.')
    else
        f3 = f3(zeros(Nx,1));
        f4 = f4(zeros(Nx,1));
    end
    
    lin_intercept = gg;
    lin_coefs = jacobian(gg,xt.');
    g0 = fct_myfun(lin_intercept,{tx}) ;
	if exist('stripstart','var') == 1
        g1 = matlabFunction(lin_coefs(:,1:stripstart-1),'vars',{tx});
        g1 = @(in) [g1(in) zeros(Nz,Ny-stripstart+1)];
    else
        g1 = matlabFunction(lin_coefs(:,1:Ny),'vars',{tx});
    end
    g2 = matlabFunction(lin_coefs(:,end-Nz+1:end),'vars',{tx});
    
    MODEL.system.f0 = f0 ;
    MODEL.system.f1 = f1 ;
    MODEL.system.f2 = f2 ;
    MODEL.system.f3 = f3 ;
    MODEL.system.f4 = f4 ;
    MODEL.system.g0 = g0 ;
    MODEL.system.g1 = g1 ;
    MODEL.system.g2 = g2 ;
    MODEL.nonlinear.f3 = f3 ;
    MODEL.nonlinear.f4 = f4 ;
    
    % z loadings on epsilon
    for i=1:length(yt)
        lambdaz = subs(lambdaz,yt(i),yt(i)+Psi_tilde(i,:)*(zt-z_tilde_aux)); % if lambdaz contains yt, replace it with its guessed solution, so there's an extra derivative wrt zt
    end
    lambdaz = sym(subs(lambdaz,params,params_val));
    sigmaz = sym(subs(sigmaz,params,params_val));
    z_loading_on_epsilon = (eye(Nz)-lambdaz*Psi_tilde)\sigmaz ;

    disp('... characterizing entropy...')
    % Entropy terms
	Nu = CCDF((f3*Psi_tilde+f4)*((eye(Nz)-lambdaz*Psi_tilde)\sigmaz));
    Nu0 = Nu;
    Nu1 = jacobian(Nu,zt.');
    Nu0 = subs(Nu0,z_tilde_aux,z_tilde);
    Nu1 = subs(Nu1,z_tilde_aux,z_tilde);
    lambdaz = subs(lambdaz,z_tilde_aux,z_tilde);
    sigmaz = subs(sigmaz,z_tilde_aux,z_tilde);
    z_loading_on_epsilon = subs(z_loading_on_epsilon,z_tilde_aux,z_tilde);
	Nu0 = fct_myfun(Nu0,{tildevar});
    Nu1 = fct_myfun(Nu1,{tildevar});

    MODEL.system.Nu0 = Nu0 ;
    MODEL.system.Nu1 = Nu1 ;
    
    MODEL.system.lambdaz = lambdaz ;
    MODEL.system.sigmaz = sigmaz ;
    
    MODEL.nonlinear.z_loading_on_epsilon = z_loading_on_epsilon ;
    