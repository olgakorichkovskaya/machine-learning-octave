function Z = projectData(X, U, K)
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.

Z = zeros(size(X, 1), K);
for i=1:size(X, 1)
x = X(i, :)';
for k=1:K
projection_k = x' * U(:, k);
Z(i,k)=projection_k;
end

end
end
