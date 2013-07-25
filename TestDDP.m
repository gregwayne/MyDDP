A           = eye(2) + randn(2,2)/sqrt(2);
B           = randn(2,2)/sqrt(2);

alpha       = 1e-1;

CostFn      = @(y) sum(y(1:2).^2) + alpha*sum(y(3:4).^2);
FinalFn     = @(y) sum(y(1:2).^2);

dt          = 0.01;
ForwardFn   = @(y) y(1:2) + dt*(tanh(A*y(1:2)) + B*y(3:4));

T           = 100;
X           = zeros(2,T);
X(:,1)      = randn(2,1);
U           = zeros(2,T);

parameters              = struct;
parameters.mu           = 1;
parameters.min_prog     = 0.1;
parameters.max_iters    = 100;
parameters.max_fails    = 10;

[V,X,U,iters]           = MyDDP(ForwardFn,CostFn,FinalFn,X,U,parameters);
disp(sprintf('Completed in %d iterations.',iters));

% Compute Null Control Trajectory
Xp                      = X;
for t=1:T-1
   
    Xp(:,t+1) = ForwardFn([Xp(:,t);0;0]);
    
end

close all;
plot(X(1,:),X(2,:),'r');
hold on;
plot(Xp(1,:),Xp(2,:),'b');
hold off;




