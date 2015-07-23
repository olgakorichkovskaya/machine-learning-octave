function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));


% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
n=size(theta);
grad = zeros(size(theta));
h=sigmoid(X*theta);
tmp=-1.*y.*log(h) - (1-y).*log(1-h);
tmp2=sum(theta(2:n).^2)* lambda/(2*m); 
J=sum(tmp)/m+tmp2;


%for i=1:size(theta)
%grad(i)=X(:,i)'*(h-y);
%end
%grad=grad/m;
%grad = (X'*(h-y))/m;

%grad = (X'*(h-y) + lambda*theta )/m;
grad(1)=(X(:,1)'*(h-y));
grad(2:end) = (X(:,2:end)'*(h-y) +lambda*theta(2:end));

grad=grad/m;




% =============================================================

grad = grad(:);

end
