function [cN,XN,UN]     = ForwardPass(ForwardFn,CostFn,FinalFn,...
                                    XN,UN,oG,cG,parameters)

        alpha   = parameters.alpha;
        T       = size(XN,2);
        
        X       = XN;
                
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

end