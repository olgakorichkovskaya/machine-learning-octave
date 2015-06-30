function J = computeCost(X, y, theta)
m = length(y); % number of training examples

J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
for iter = 1:m
J=J+(theta(1,1) +theta(2,1)*X(iter,2)-  y(iter))^2;
end

% =========================================================================
J=J/(2*m);
end
