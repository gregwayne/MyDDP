function [cN,XN,UN,iters] = DDP(ForwardFn,CostFn,FinalFn,X,U,parameters)
    %% Based on Tom Erez's Ph.D. Thesis.
    %% Implemented by Greg Wayne, 2013.

    NX          = size(X,1);
    NU          = size(U,1);
    T           = size(X,2);
    
    CostRate    = @(Y) CostFn(Y)/T;

    oG          = zeros(size(U));
    cG          = zeros([NU,NX,T]);   
        
    %% Initialization
    [c,X,U]     = ForwardPass(ForwardFn,CostRate,FinalFn,...
                                    X,U,oG,cG,parameters);
    
    iters       = 1;
    fails       = 0;
    local_min   = 0;
    
    while true
                       
        if (iters >= parameters.max_iters) ...
                || (fails > parameters.max_fails) ...
                || local_min
            break;
        end
        
        %% Dynamics

        [f_der,l_der]               = DynamicsStep(ForwardFn,CostRate,X,U);

        %% Backward Pass
        [cG,oG,dVs_lin,dVs_quad]    = BackwardPass(FinalFn,X,U,...
                                            f_der,l_der,parameters);

        parameters.alpha            = 1;
        
        disp(sprintf('Iteration %d',iters));

        %% Forward Pass
        while true

            [cN,XN,UN]  = ForwardPass(ForwardFn,CostRate,FinalFn,...
                                        X,U,oG,cG,parameters);
            
            dV          = parameters.alpha*sum(dVs_lin) ...
                            + parameters.alpha^2*sum(dVs_quad);
                        
            if abs(sum(dV)) < parameters.tol
                local_min   = 1;
                break;
            end
            
            z           = (cN - c)/dV;
    
            if z >= 0.5
                
                disp('Armijo-like Condition Met');
                disp(sprintf('Cost == %f',cN));
                                
                X       = XN;
                U       = UN;
                c       = cN;
                fails   = 0;
                break;
                                
            else
                
                disp('Armijo-like Condition Not Met');
                fails            = fails + 1;
                parameters.alpha = parameters.alpha/2;
                
            end
                            
        end

        iters = iters + 1;
    
    end
    
end