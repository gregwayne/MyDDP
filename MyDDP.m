function [c,X,U,iters] = MyDDP(ForwardFn,CostFn,FinalFn,X,U,parameters)

    XN          = X;
    UN          = U;
    
    NX          = size(X,1);
    NU          = size(U,1);
    T           = size(X,2);
        
    oG          = zeros(size(U));
    cG          = zeros([NU,NX,T]);
        
    mu          = parameters.mu;
    max_iters   = parameters.max_iters;
    min_prog    = parameters.min_prog;
    max_fails   = parameters.max_fails;
    c           = inf; 
    fails       = 0;
    iters       = 0;    
    
    while true
        
        iters   = iters + 1;        
        cN      = 0;
        %% Forward Pass
        for t=1:T-1
            
            UN(:,t)     = UN(:,t) ...
                          + oG(:,t) + cG(:,:,t)*(XN(:,t) - X(:,t));
            XN(:,t+1)   = ForwardFn([XN(:,t);UN(:,t)]);           
            cN          = cN + CostFn([XN(:,t);UN(:,t)]);
            
        end        
        
        % Final Cost
        cN              = cN + FinalFn(XN(:,T)); 
        
        if (cN < c - min_prog)
            c       = cN;
            X       = XN;
            U       = UN;
            mu      = 0.8*mu;
            fails   = 0;
        else
            cN      = c;
            XN      = X;
            UN      = U;
            mu      = 2*mu;
            fails   = fails + 1;
        end
            
        disp(sprintf('Trajectory Cost: %f',cN));
        
        if (iters > max_iters) || (fails > max_fails)        
            break;            
        end          
        
        % Initialize Backward Pass
        RegFinalFn  = @(xT) FinalFn(xT) + mu*sum((xT - X(:,T)).^2);
        [V,Vx,Vxx]  = AllDerivatives(RegFinalFn,XN(:,T));
                
        %% Backward Pass
        for t=T-1:-1:1
                        
            % Basic Derivatives
            [f,fx,fu,fxx,fxu,fuu] = ParseDerivatives(ForwardFn,...
                                       [XN(:,t);UN(:,t)],NX,NU);
                                   
            RegCostFn = @(z) CostFn(z) + mu*sum((z(1:NX) - X(:,t)).^2);
                                   
            [l,lx,lu,lxx,lxu,luu] = ParseDerivatives(RegCostFn,...
                                       [XN(:,t);UN(:,t)],NX,NU);
                                   
            lxx                 = reshape(lxx,[NX,NX]);                                   
            lxu                 = reshape(lxu,[NX,NU]);                                   
            luu                 = reshape(luu,[NU,NU]);                                   
                                               
            % Hamiltonian Terms
            Q                   = l + V;
            Qx                  = lx + Vx*fx;
            Qu                  = lu + Vx*fu;
            % Before contraction term
            Qxx                 = lxx + fx'*Vxx*fx;
            Qxu                 = lxu + fx'*Vxx*fu;
            Quu                 = luu + fu'*Vxx*fu;
            
            for i=1:NX
               
                Qxx     = Qxx + reshape(Vx(i)*fxx(i,:,:),[NX,NX]);
                Qxu     = Qxu + reshape(Vx(i)*fxu(i,:,:),[NX,NU]);
                Quu     = Quu + reshape(Vx(i)*fuu(i,:,:),[NU,NU]);
                
            end                    
                
            IQuu                = pinv(Quu);            
            oG(:,t)             = -IQuu*Qu';
            cG(:,:,t)           = -IQuu*Qxu';
            
            % Value Terms
            V                   = Q - Qu*IQuu*Qu';
            Vx                  = Qx - Qu*IQuu*Qxu';
            Vxx                 = Qxx - Qxu*IQuu*Qxu';
                        
        end                        
        
    end
    
end