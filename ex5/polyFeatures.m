function [X_poly] = polyFeatures(X, p)
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
X_poly = zeros(numel(X), p);

n=size(X,2);
X_poly(:,1:n) = X(:,:);
for i = 1:p-1
      X_poly(:, n+i) = (X.^(i+1));
end

end
