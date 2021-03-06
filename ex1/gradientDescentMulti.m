function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n=length(theta);

t_history = zeros(num_iters, n);
t=zeros(n,1);
for iter = 1:num_iters
for i=1:n
  %  sum=sum1(X, y, theta,m, i,n);
   
t(i,1)=theta(i,1)-alpha*sum( (X*theta-y)'*X(:,i))/m;
end
theta=t;
    % Save the cost J in every iteration    
    curent_J= computeCostMulti(X, y, theta);
    J_history(iter) =  curent_J;
    
    
t_history(iter,:) = theta(:,1);

end
plotData1( 1:num_iters, J_history,'iteration','J');

tt= computeCost(X, y, theta); 
tml='Best J %.3f \n';
fprintf(tml,tt);
   
end
%==========================================
function J = sum1(X, y, theta, m, j, n)

div=X*theta-y;
J=sum( div'*X(:,j))/m;

%for iter = 1:m
%tmp=0;
%for i=1:n
%    tmp=tmp+theta(i,1)*X(iter,i);
%end
%tmp=tmp- y(iter);
%J=J+tmp*X(iter,j);
%end
%J=J/m;
end