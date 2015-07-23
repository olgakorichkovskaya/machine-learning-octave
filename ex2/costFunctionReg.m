function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); % number of training examples

n=size(theta);
grad = zeros(n);
h=sigmoid(X*theta);
tmp=-1.*y.*log(h) - (1-y).*log(1-h);

tmp2=sum(theta(2:n).^2)* lambda/(2*m); 
J=sum(tmp)/m+tmp2;

grad(1)=(X(:,1)'*(h-y));


%for j=2:size(theta)
%grad(j)=X(:,j)'*(h-y)+lambda*theta(j);
%end

grad(2:end) = (X(:,2:end)'*(h-y) +lambda*theta(2:end));

%for j=2:size(theta)
%grad(j)=grad(j)+lambda*theta(j);
%end
grad=grad/m;
end
