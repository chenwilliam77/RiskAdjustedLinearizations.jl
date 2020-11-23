% fct_solution.m is a routine that solves the system for a generalized
% affine approximation to the solution of a DSGE.
%
% Called by: genaffine.m
%
% Written by Pierlauro Lopez and Francisco Vazquez-Grande.
% (c) This version: October 2017.

    tz = MODEL.variables.z ;
    ty = MODEL.variables.y ;
    tx = MODEL.variables.x ;
    Psi = MODEL.policy.Psi_tilde;
    tPsi = Psi(:) ;
	tildevar = [ty;tz;tPsi] ;
    Ny = length(ty) ;
    Nz = length(tz) ;
    Nx = Ny + Nz ;

    % faster, using sparse matrices
	f0 = MODEL.system.f0;
	f1 = MODEL.system.f1;
    f2 = MODEL.system.f2;
	f3 = MODEL.system.f3;
    f4 = MODEL.system.f4;
    Nu0 = MODEL.system.Nu0;
    Nu1 = MODEL.system.Nu1;
	g0 = MODEL.system.g0;
	g1 = MODEL.system.g1;
    g2 = MODEL.system.g2;


%% Solution at Deterministic SS

    % tic;
    % disp('... 1st order approximation around the deterministic steady state...')

    expr0_dss = @(in) [f0(in) + f3*in(1:Ny) + f4*in(Ny+1:Ny+Nz); g0(in)-in(Ny+1:Ny+Nz)] ;
    options = optimset('tolx',1d-15,'tolfun',1d-15,'maxfunevals',1d7,'maxiter',1d7,'display','off','LargeScale','on');
    try
        x_dss = fsolve(expr0_dss,MODEL.calibration.DSS,options);
        x_dss = real(x_dss); if max(abs(expr0_dss(x_dss))) > 1d-10; error('imaginary solution'); end
    catch
        try
            x_dss = fsolve(expr0_dss,zeros(length(tx),1),options);
            x_dss = real(x_dss); if max(abs(expr0_dss(x_dss))) > 1d-10; error('imaginary solution'); end
        catch
            warning('initial condition to solve for deterministic steady state is too far from the solution')
            x_dss = fsolve(expr0_dss,ones(length(tx),1),options);
        end
    end

    f0_num = full(f0(x_dss));
    f1_num = full(f1(x_dss));
    f2_num = full(f2(x_dss));
    f3_num = full(f3);
    f4_num = full(f4);
    g0_num = full(g0(x_dss));
    g1_num = full(g1(x_dss));
    g2_num = full(g2(x_dss));
    Psi_dss = qzdecomp(f1_num,f2_num,f3_num,f4_num,g1_num,g2_num) ;

    sol_dss = [x_dss;Psi_dss(:)];

    x_dss = sol_dss(1:Nx) ;
    y_dss = x_dss(1:Ny) ;
    z_dss = x_dss((Ny+1):end) ;
    Psi_dss = reshape(sol_dss((Nx+1):end),Ny,Nz) ;

    MODEL.solution.dss.x = x_dss ;
    MODEL.solution.dss.y = y_dss ;
    MODEL.solution.dss.z = z_dss ;
    MODEL.solution.dss.Psi = Psi_dss ;
    MODEL.solution.dss.f0 = f0_num ;
    MODEL.solution.dss.f1 = f1_num ;
    MODEL.solution.dss.f2 = f2_num ;
    MODEL.solution.dss.f3 = f3_num ;
    MODEL.solution.dss.f4 = f4_num ;
    MODEL.solution.dss.g0 = g0_num ;
    MODEL.solution.dss.g1 = g1_num ;
    MODEL.solution.dss.g2 = g2_num ;

    MODEL.solution.dss.zloading = double(subs(subs(MODEL.nonlinear.z_loading_on_epsilon,MODEL.policy.Psi_tilde(:),Psi_dss(:)),xt,x_dss)) ;

    % disp('... solved!')
    % toc;

%% Solution at Risky Steady State.

    % tic;
    % disp('... 1st order approximation around the risky steady state...')

    x0 = x_dss ;
    Psi0 = Psi_dss ;
    sol0 = [x0;Psi0(:)] ;

    % disp('Choose solution algorithm: (=0 full scale homotopy search; =1 iterations on QZ decomposition)')
    % sol_opt = input('');
    time_algo = tic();
    if sol_opt == 0
        % disp('Pick homotopy continuation step in (0,1] (1 means no continuation)')
        % step = input('');
        step = .1;
        if step <= 0 || step > 1
            error('Homotopy continuation step must be in (0,1]')
        end
        sol = sol0;
        for i=step:step:1
            expr0 = @(in) [f0(in) + f3*in(1:Ny) + f4*in(Ny+1:Ny+Nz) + i*Nu0(in); g0(in)-in(Ny+1:Ny+Nz)] ;
            expr1 = @(in) f1(in)*reshape(in(Nx+1:end),Ny,Nz)+f2(in)+(f3*reshape(in(Nx+1:end),Ny,Nz)+f4)*(g1(in)*reshape(in(Nx+1:end),Ny,Nz)+g2(in)) + i*Nu1(in);
            expr = @(in) [expr0(in);vec(expr1(in))];
            sol = fsolve(expr,sol,options);
            % disp(['progress = ' num2str(i)])
        end
    else
        sol_new = sol0; sol = sol0;
        err = 1; count = 0;
        while err > 1d-10
            sol = .5*sol_new+.5*sol;
            count = count+1;
            x_sol = sol(1:Nx);
            Nu0_ = Nu0(sol);
            Nu1_ = Nu1(sol);
            expr0 = @(in) [f0(in) + [f3 f4]*in(1:Nx) + Nu0_; g0(in)-in(Ny+1:Ny+Nz)] ;
            x_sol = fsolve(expr0,x_sol,options);
            Psi_sol = qzdecomp(f1(x_sol),f2(x_sol)+Nu1_,f3,f4,g1(x_sol),g2(x_sol),0) ;
            sol_new = [x_sol;Psi_sol(:)];
            err = max(abs(sol_new-sol));
            % disp(['iteration: ' num2str(count) '; error: ' num2str(err)])
        end
    end
    toc(time_algo);
    sol_rss = sol ;

    x_rss = sol_rss(1:Nx) ;
    y_rss = x_rss(1:Ny) ;
    z_rss = x_rss((Ny+1):end) ;
    Psi_rss = reshape(sol_rss((Nx+1):end),Ny,Nz) ;

    f0_num = full(f0(x_rss));
    f1_num = full(f1(x_rss));
    f2_num = full(f2(x_rss));
    f3_num = full(f3);
    f4_num = full(f4);
    g0_num = full(g0(x_rss));
    g1_num = full(g1(x_rss));
    g2_num = full(g2(x_rss));
    Nu0_num = full(Nu0(sol_rss));
    Nu1_num = full(Nu1(sol_rss));
    Gamma = [f4_num f3_num;eye(Nz) zeros(Nz,Ny)];
    Xi = [-f2_num-Nu1_num -f1_num;g2_num g1_num];
    if sum(abs(eig(Gamma,Xi)) > 1) ~= Nz
        warning('1st-order perturbation around risky steady state is not saddle-path stable')
    else
        % disp('LLV conditions for a unique locally bounded risky steady-state perturbation are satisfied')
    end

    MODEL.solution.rss.x = x_rss ;
    MODEL.solution.rss.y = y_rss ;
    MODEL.solution.rss.z = z_rss ;
    MODEL.solution.rss.Psi = Psi_rss ;
    MODEL.solution.rss.f0 = f0_num ;
    MODEL.solution.rss.f1 = f1_num ;
    MODEL.solution.rss.f2 = f2_num ;
    MODEL.solution.rss.f3 = f3_num ;
    MODEL.solution.rss.f4 = f4_num ;
    MODEL.solution.rss.g0 = g0_num ;
    MODEL.solution.rss.g1 = g1_num ;
    MODEL.solution.rss.g2 = g2_num ;
    MODEL.solution.rss.Nu0 = Nu0_num ;
    MODEL.solution.rss.Nu1 = Nu1_num ;

	zloading_fun_rss = matlabFunction(subs(MODEL.nonlinear.z_loading_on_epsilon,MODEL.policy.Psi_tilde(:),Psi_rss(:)),'vars',{xt}) ;
    MODEL.solution.rss.zloading_fun = zloading_fun_rss ;

    % disp('... solved!')
    % toc;
