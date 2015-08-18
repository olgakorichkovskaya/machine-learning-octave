function [mu sigma2] = estimateGaussian(X)
% function estimates the parameters of a 
% Gaussian distribution using the data in X
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector

[m, n] = size(X);
mu = zeros(n, 1);
sigma2 = zeros(n, 1);
for i =1:n
 x=X(:,i);
 mem = mean(x);
 s=sum(((x - mem)).^2)/m;
 
 mu(i,1)=mem;
 sigma2(i,1)=s;
end

end
