function [f,df,ddf] = AllDerivatives(fn,x,varargin)
    
    [df,f]  = FD(fn,x);
    df      = permute(df,numel(size(df)):-1:1);
    ddf     = FD(@(y) FD(fn,y), x);   
    ddf     = permute(ddf,numel(size(ddf)):-1:1);
    
end