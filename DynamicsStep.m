function [f_der,l_der] = DynamicsStep(ForwardFn,CostRate,XN,UN)

    NX          = size(XN,1);
    NU          = size(UN,1);
    T           = size(XN,2);

    f_der       = {};
    l_der       = {};
    
    for t=1:T

        % Basic Derivatives
        [f,fx,fu,fxx,fxu,fuu]   = ParseDerivatives(ForwardFn,...
                                   [XN(:,t);UN(:,t)],NX,NU);
                               
        f_der{t}                = {f,fx,fu,fxx,fxu,fuu};
        
        [l,lx,lu,lxx,lxu,luu]   = ParseDerivatives(CostRate,...
                                   [XN(:,t);UN(:,t)],NX,NU);

        l_der{t}                = {l,lx,lu,lxx,lxu,luu};
        
    end

end