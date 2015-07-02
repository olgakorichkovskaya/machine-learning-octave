function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));
h=sigmoid(X*theta);
tmp=-1.*y.*log(h) - (1-y).*log(1-h);
J=sum(tmp)/m;

for i=1:size(theta)
grad(i)=X(:,i)'*(h-y);
end
grad=grad/m;

end
