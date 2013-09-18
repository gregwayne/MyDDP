function [cG,oG,dVs_lin,dVs_quad] = BackwardPass(FinalFn,XN,UN,...
                                            f_der,l_der,parameters)

    NX          = size(XN,1);
    NU          = size(UN,1);
    T           = size(XN,2);
        
    oG          = zeros(size(UN));
    cG          = zeros([NU,NX,T]);

    dVs_lin     = zeros(T,1);
    dVs_quad    = zeros(T,1);
        
    parameters.lambda = 1e-2;
            
    [V,Vx,Vxx]  = AllDerivatives(FinalFn,XN(:,T));
             
    %% Backward Pass
    for t=T-1:-1:1

        success     = 0;
        
        gamma       = 1;
        lambda      = parameters.lambda;
        
        while (success == 0)
            % Basic Derivatives

            [f,fx,fu,fxx,fxu,fuu] = deal(f_der{t}{:});

            [l,lx,lu,lxx,lxu,luu] = deal(l_der{t}{:});

            lxx         = reshape(lxx,[NX,NX]); 
            lxx         = lxx + lambda*eye(NX,NX);
            lxu         = reshape(lxu,[NX,NU]);                                   
            luu         = reshape(luu,[NU,NU]);
            luu         = luu + parameters.ratio*lambda*eye(NU,NU);

            % Hamiltonian Terms
            Q                   = l + V;
            Qx                  = lx + Vx*fx;
            Qu                  = lu + Vx*fu;

            % Before contraction terms
            Qxx                 = lxx + fx'*Vxx*fx;
            Qxu                 = lxu + fx'*Vxx*fu;
            Quu                 = luu + fu'*Vxx*fu;

            % Contraction terms
            for i=1:NX

                Qxx     = Qxx + reshape(Vx(i)*fxx(i,:,:),[NX,NX]);
                Qxu     = Qxu + reshape(Vx(i)*fxu(i,:,:),[NX,NU]);
                Quu     = Quu + reshape(Vx(i)*fuu(i,:,:),[NU,NU]);

            end         
                        
            [E,D]               = eig(Quu);
            if min(diag(D)) > 0
                success     = 1;
            else
                disp('Not Positive Definite');
                gamma       = gamma*5;
                lambda      = lambda*gamma; 
                break;
            end

            IQuu                = E*diag(1./diag(D))*E';            
            
            oG(:,t)             = -IQuu*Qu';
            cG(:,:,t)           = -IQuu*Qxu';

            % Value Terms
            V                   = Q - Qu*IQuu*Qu';
            Vx                  = Qx - Qu*IQuu*Qxu';
            Vxx                 = Qxx - Qxu*IQuu*Qxu';

            dVs_lin(t)          = oG(:,t)'*Qu';
            dVs_quad(t)         = oG(:,t)'*Quu*oG(:,t)/2;

        end
                
    end            
        
end
