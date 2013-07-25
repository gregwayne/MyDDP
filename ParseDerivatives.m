function [f,fx,fu,fxx,fxu,fuu] = ParseDerivatives(fn,z,NX,NU)

    [f,fz,fzz] = AllDerivatives(fn,z);
    
    if size(fzz,2) ~= (NX + NU)
        
        error('Size of combined derivative is not same as state-control size.');
    
    end
    
    fx  = fz(:,1:NX);
    fu  = fz(:,NX+1:end);
 
    if numel(size(fzz)) == 3
    
        fxx = fzz(:,1:NX,1:NX);
        fxu = fzz(:,1:NX,NX+1:end);
        fuu = fzz(:,NX+1:end,NX+1:end);
    
    else

        fxx = fzz(1:NX,1:NX);
        fxu = fzz(1:NX,NX+1:end);
        fuu = fzz(NX+1:end,NX+1:end);
        
    end
    
    NF      = size(f,1);
    fxx     = reshape(fxx,[NF,NX,NX]);
    fxu     = reshape(fxu,[NF,NX,NU]);
    fuu     = reshape(fuu,[NF,NU,NU]);
    
end