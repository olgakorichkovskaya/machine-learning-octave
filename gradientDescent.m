function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n=length(theta);

t_history = zeros(num_iters, n);
best_j=9999999;
best_t=zeros(n,1);
t=zeros(n,1);
for iter = 1:num_iters

for i=1:n
t(i,1)=theta(i,1)-alpha*sum1(X, y, theta,m, i,n);
end
theta=t;
    % Save the cost J in every iteration    
    %J_history(iter) = computeCost(X, y, theta);
    J_history(iter) = computeCostMulti(X, y, theta);
    
t_history(iter,:) = theta(:,1);
if(J_history(iter)< best_j)
    best_j = J_history(iter);
    best_t=t;
end

end
plotData1( 1:num_iters, J_history,'iteration','J');
%plotData2Dim( 1:num_iters, t_history(:,1), t_history(:,2), 'iteration','theta 0','theta 1');

%theta=best_t;
tt= computeCost(X, y, theta); 
tml='Best J %.3f \n';
fprintf(tml,tt);
   
end
%==========================================
function J = sum1(X, y, theta, m, j, n)
J = 0;

for iter = 1:m
tmp=0;
for i=1:n
    tmp=tmp+theta(i,1)*X(iter,i);
end
tmp=tmp- y(iter);
J=J+tmp*X(iter,j);
end
J=J/m;
end