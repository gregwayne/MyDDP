% [Df,fx] = FD(f,x,eps)
%
% Finite Differences based on Forward Differences, not Central.
% Returns multiple values: first the derivatives then the function value 
% to allow for simple computation of both gradients and Hessians.
%
% FD(fn,x) or FD(fn, x, eps) is the gradient and
% FD(@(y) FD(fn, y), x) is the Hessian.
% 
% Greg Wayne, 2013
function [Df,fx] = FD(f,x,varargin)

    if nargin == 2
       
        eps = 1e-4;
        
    else
        
        eps = varargin{1};
        
    end

    fx  = f(x);
    nx  = numel(x);
    Df  = zeros([nx,size(fx)]);    
    nd  = numel(size(Df));
    ey  = eps*eye(nx);
        
    for i=1:nx
        
        a       = (f(x + ey(:,i)) - fx)/eps;
        Df(i,:) = a(:);        
        
    end
        
end