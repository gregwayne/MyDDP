function [c,X,U,iters] = MyDDP(ForwardFn,CostFn,FinalFn,X,U,parameters)

    XN          = X;
    UN          = U;
    
    NX          = size(X,1);
    NU          = size(U,1);
    T           = size(X,2);
        
    oG          = zeros(size(U));
    cG          = zeros([NU,NX,T]);
        
    mu          = parameters.mu;
    alpha       = parameters.alpha;    
    max_iters   = parameters.max_iters;
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
                          + alpha*oG(:,t) + cG(:,:,t)*(XN(:,t) - X(:,t));
            XN(:,t+1)   = ForwardFn([XN(:,t);UN(:,t)]);           
            cN          = cN + CostFn([XN(:,t);UN(:,t)]);
            
        end
        
        % Final Cost
        cN              = cN + FinalFn(XN(:,T));        
        
        if (cN >= c)
            % revert
            mu          = 2*mu;            
            alpha       = max(0.9*alpha,0.01);            
            XN          = X;            
            UN          = U;
            cN          = c;
            fails       = fails + 1;
        else
            % move on
            mu          = 0.5*mu; % reduce regularizer
            alpha       = min(1.1*alpha,1);
            X           = XN;
            U           = UN;
            fails       = 0;
        end                
        
        c               = cN;
        
        disp(sprintf('Trajectory Cost: %f',cN));
        mu,alpha
        
        % Initialize Backward Pass
        [~,Vx,Vxx]  = AllDerivatives(FinalFn,XN(:,T));
        % regularization
        Vx          = Vx + mu*(XN(:,T) - X(:,T))';
        Vxx         = Vxx + mu*eye(NX);
                
        %% Backward Pass
        for t=T-1:-1:1
                        
            % Basic Derivatives
            [fx,fu,fxx,fxu,fuu] = ParseDerivatives(ForwardFn,...
                                       [XN(:,t);UN(:,t)],NX,NU);
                                           
            [lx,lu,lxx,lxu,luu] = ParseDerivatives(CostFn,...
                                       [XN(:,t);UN(:,t)],NX,NU);
                                   
            lxx                 = reshape(lxx,[NX,NX]);                                   
            lxu                 = reshape(lxu,[NX,NU]);                                   
            luu                 = reshape(luu,[NU,NU]);                                   
                                   
            % regularization           
            lx                  = lx + mu*(XN(:,t) - X(:,t))';
            lxx                 = lxx + mu*eye(NX);
            
            % Hamiltonian Derivatives
            Qx                  = lx + Vx*fx;
            Qu                  = lu + Vx*fu;
            % Before contraction term
            Qxx                 = lxx + fx'*Vxx*fx;
            Qxu                 = lxu + fx'*Vxx*fu;
            Quu                 = luu + fu'*Vxx*fu;
            
            for i=1:NX
               
                Qxx = Qxx + reshape(Vx(i)*fxx(i,:,:),[NX,NX]);
                Qxu = Qxu + reshape(Vx(i)*fxu(i,:,:),[NX,NU]);
                Quu = Quu + reshape(Vx(i)*fuu(i,:,:),[NU,NU]);
                
            end
                    
            % Control Update
            try
                [E,D]               = eig(Quu);                
                D                   = max(D,1);
                DQuu                = E*D*E';
            catch errorMsg,
                DQuu                = 10*eye(NU);
            end
                
            IQuu                = pinv(DQuu);            
            oG(:,t)             = -IQuu*Qu';
            cG(:,:,t)           = -IQuu*Qxu';
            
            % Value Derivatives
            Vx                  = Qx - Qu*IQuu*Qxu';
            Vxx                 = Qxx - Qxu*IQuu*Qxu';
                        
        end        
        
        % Termination Criterion
        if (iters > max_iters) || (fails >= max_fails)            
            break;            
        end        
            
    end
    
end