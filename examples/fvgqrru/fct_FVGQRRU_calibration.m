function [ MODEL ] = fct_FVGQRRU_calibration( MODEL, Country, MM )
%[ MODEL ] = model_calibration_fct( MODEL, Country, MM )
%   Adds the calibration field

    % Common Parameters
    nu          = 5 ;
    eta         = 1000 ;
    delta       = 0.014 ;
    alpha       = 0.32 ;
    rhoxx       = .95 ;
    omega       = 1 ; % made up not in the paper
    
    switch MM
        case 1
            % T-bill (table 3)
            rhotb       = .95 ;
            sigmatb     = -8.05 ;
            rhosigmatb  = .94 ;
            etatb       = .13 ;
        case 2
            % T-bill (table 4)
            rhotb       = .95 ;
            sigmatb     = -8.05 ;
            rhosigmatb  = .94 ;
            etatb       = .13 ;
        otherwise
            disp('M not found')
            exit
    end
    
    
    switch Country
        case 'Argentina'
            if MM==1
                % Argentina M1 (table 3)
                rhor        = .97 ;
                sigmar      = -5.71 ;
                rhosigmar   = .94 ;
                etar        = .46 ;
                kappa       = 0 ;
                % Argentina M1 (table 6)
                beta        = .980 ;
                r           = -log(beta) ;
                PhiD        = .001 ;
                D           = 4 ;
                phi         = 95 ;
                sigmaxx     = .015 ;
            elseif MM==2
                % Argentina M2 (table 4)
                rhor        = .97 ;
                sigmar      = -5.80 ;
                rhosigmar   = .90 ;
                etar        = .45 ;
                kappa       = .69 ;
                % Argentina M2 (table 6)
                beta        = .980 ;
                r           = -log(beta) ;
                PhiD        = .001 ;
                D           = 4 ;
                phi         = 85 ;
                sigmaxx     = .014 ;
            else
                disp('M not found')
                exit
            end
        case 'Ecuador'
            if MM==1
                % Ecuador M1 (table 3)
                rhor        = .95 ;
                sigmar      = -6.06 ;
                rhosigmar   = .96 ;
                etar        = .35 ;
                kappa       = 0 ;
                % Ecuador M1 (table 6)
                beta        = .989 ;
                r           = -log(beta) ;
                PhiD        = .001 ;
                D           = 13 ;
                phi         = 35 ;
                sigmaxx     = .0055 ;
            elseif MM==2
                % Ecuadora M2 (table 4)
                rhor        = .95 ;
                sigmar      = -5.93 ;
                rhosigmar   = .89 ;
                etar        = .34 ;
                kappa       = .89 ;
                % Ecuador M2 (table 6)
                beta        = .989 ;
                r           = -log(beta) ;
                PhiD        = .001 ;
                D           = 13 ;
                phi         = 20 ;
                sigmaxx     = .0058 ;
            else
                disp('M not found')
                exit
            end
        otherwise
            disp('country not found')
    end
    
    tic;
    disp('---------------------Parametrizing---------------------')
    Npar = length(MODEL.parameters.params);
    for i=1:Npar;
        eval(['params(i,1) = ' char(MODEL.parameters.params(i)) ';'])
    end
    
    MODEL.calibration.params = params ;
    MODEL.calibration.Country = Country ;
    MODEL.calibration.MM = MM ; 
    
	disp('... done!')
    toc;
    
end

