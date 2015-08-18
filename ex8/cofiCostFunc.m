function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
           
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

a= X*Theta' -Y;
aa=R .*(a .^2);
userNum=size(R,2);
featureNum=size(X,2);
movieNum=size(R,1);
sumTheta=0;
sumX=0;
for k=1:featureNum
 for j=1:userNum;
  sumTheta=sumTheta+Theta(j,k)^2;
 end
 
 for i=1:movieNum
  sumX=sumX+X(i,k)^2;
 end    
end 

%                               regularization
J=sum(sum(aa)) +lambda*sumTheta+lambda*sumX;
J=J/2;

%movies x features
for i=1:size(R,1)
idx = find(R(i, :)==1);
Thetatemp = Theta(idx, :);

Ytemp = Y(i, idx);
c=(X(i, :) * Thetatemp' -Ytemp);
X_grad(i, :) = c*Thetatemp+ lambda*X(i,:);
end


for j=1:userNum %4
idM = find(R(:,j)==1); % instead m - movies
Ytemp = Y(idM, j);     % temp x 1
Thetatemp = Theta(j,:);% 1 x f 
Xtemp = X(idM, :);     % temp x f
a=Xtemp * Thetatemp';  % tempM x 1
b = (a - Ytemp);       % tempM x 1

%users x features            regularization
Theta_grad(j,:) =b' * Xtemp +lambda*Theta(j,:);
end


% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
grad = [X_grad(:); Theta_grad(:)];

end
