function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
h= X*theta;
%regularization
sum2=0;
for j=2:size(theta,1)
     sum2=sum2+theta(j,1)^2;
end

J= sum((h-y).^2) +lambda* sum2;
J=J/(2*m);

grad(1)=(X(:,1)'*(h-y));
grad(2:end) = (X(:,2:end)'*(h-y) +lambda*theta(2:end));

grad=grad/m;
grad = grad(:);

end
