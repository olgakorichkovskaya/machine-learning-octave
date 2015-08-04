function [U, S] = pca(X)
[m, n] = size(X);

U = zeros(n);
S = zeros(n);
%covariance matrix
Sigma=X'*X/m;
%eigenvectors  and eigenvalues of the covariance matrix. 
[U, S, V] = svd(Sigma);
end
