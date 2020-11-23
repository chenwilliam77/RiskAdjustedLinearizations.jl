function [f_y,f_z] = fct_myfun(f,tildevar,vinfo)

    if nargin < 3
        
        f_y = matlabFunction(f,'vars',tildevar); 
        f_z = [];
                        
    else
        
        if length(vinfo) == 2

            Ny = vinfo(1);
            Nz = vinfo(2);
            f_y = matlabFunction(f(:,1:Ny),'vars',tildevar); 
            f_z = matlabFunction(f(:,Ny+1:Ny+Nz),'vars',tildevar);
        
        elseif length(vinfo) == 4

            Ns = vinfo(1);
            stripstart = vinfo(2);
            stripclasses = vinfo(3);
            sign = vinfo(4);

            [Nr,Nc] = size(f);
            f_ = matlabFunction(f,'vars',tildevar);

            f_y = @(in) fct_myfun_y(f_(in),Nr,Ns,stripstart,stripclasses,sign);  
            f_z = @(in) fct_myfun_z(f_(in),Nr,Nc,Ns,stripstart,stripclasses);  
        
        else
            
        	error('argument error: provide (i) number of strips; (ii) what row strips start; (iii) number of classes of strips; (iv) whether we are creating f1 (info=-1) or f3 (info=1)')

        end
        
    end

end
    
function A = fct_myfun_y(f_,Nr,Ns,stripstart,stripclasses,sign)

    Ny = Nr+(Ns-1-(Ns>1))*stripclasses;
    
%	A = sparse(zeros(Ny,Ny));
	A = zeros(Ny,Ny);
    A(1:stripstart-1,:) = f_(1:stripstart-1,1:Ny);

    if Ns > 1
        for i = 1:stripclasses
            A(stripstart+(i-1)*Ns,1:stripstart-1) = f_(stripstart+(i-1)*2,1:stripstart-1);
            A(stripstart+(i-1)*Ns+1:stripstart+(i-1)*Ns+Ns-1,1:stripstart-1) = kron(ones(Ns-1,1),f_(stripstart+(i-1)*2+1,1:stripstart-1));
        end
        aux1 = (-eye(Ns*stripclasses));
        aux2 = diag(kron(ones(stripclasses,1),[ones(Ns-1,1);0])); aux2(end,:) = []; aux2(:,end) = []; aux2 = [zeros(1,Ns*stripclasses-1) 0;aux2 zeros(Ns*stripclasses-1,1)];
        A(stripstart:Ny,stripstart:Ny) = (sign<0)*aux1+(sign>0)*aux2;
    else
        for i = 1:stripclasses
            A(stripstart+(i-1)*Ns,1:stripstart-1) = f_(stripstart+(i-1),1:stripstart-1);
        end
        aux1 = (-eye(Ns*stripclasses));
        aux2 = diag(kron(ones(stripclasses,1),[ones(Ns-1,1);0])); aux2(end,:) = []; aux2(:,end) = []; aux2 = [zeros(1,Ns*stripclasses-1) 0;aux2 zeros(Ns*stripclasses-1,1)];
        A(stripstart:Ny,stripstart:Ny) = (sign<0)*aux1+(sign>0)*aux2;
    end
    
end

function A = fct_myfun_z(f_,Nr,Nc,Ns,stripstart,stripclasses)

    Ny = Nr+(Ns-1-(Ns>1))*stripclasses;
    Nz = Nc-Ny;
    
%	A = sparse(zeros(Ny,Nz));
	A = zeros(Ny,Nz);    
    A(1:stripstart-1,1:Nz) = f_(1:stripstart-1,end-Nz+1:end);

	if Ns > 1
        for i = 1:stripclasses
            A(stripstart+(i-1)*Ns,1:Nz) = f_(stripstart+(i-1)*2,end-Nz+1:end);
            A(stripstart+(i-1)*Ns+1:stripstart+(i-1)*Ns+Ns-1,1:Nz) = kron(ones(Ns-1,1),f_(stripstart+(i-1)*2+1,end-Nz+1:end));
        end
    else
        for i = 1:stripclasses
            A(stripstart+(i-1)*Ns,1:Nz) = f_(stripstart+(i-1),end-Nz+1:end);
        end
    end

end