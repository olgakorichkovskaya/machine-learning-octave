function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%m = the number of training examples
%n = the number of training features, including the initial bias unit.
%h = the number of units in the hidden layer - NOT including the bias unit
%r = the number of output classifications 10


% a3, a2, z2, and a1 
a1=[ones(m, 1) X];
z2=a1 * Theta1'; % (m x n) ? (n x h) --> (m x h) 5000x25
a2 = sigmoid(z2);% 5000x25
a2=[ones(m, 1) a2];%??
a3 = sigmoid(a2 * Theta2');
%[t, p] = max(a3, [], 2);


K= num_labels;
y_mat = eye(num_labels);
y1 = y_mat(y,:);
sum=0;
for i=1:m
 for k=1:K
     add1=-y1(i,k)*log(a3(i,k));
     add2=(1-y1(i,k))*log(1-a3(i,k));
 sum=sum+add1-add2;
 end
end
J=sum/m;

d3=a3-y1;% mxr 5000x10

% d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product 
%of d3 and Theta2(no bias),
%then element-wise scaled by sigmoid gradient of z2. 
d2=d3*Theta2(:,2:end); % (m x r) ? (r x h) --> (m x h) 5000x25

% Delta1 is the product of d2 and a1
Delta1=d2'*a1; % (h x m) ? (m x n) --> (h x n) 25x 401
Delta2=d3'*a2;  % (r x m) ? (m x [h+1]) --> (r x [h+1]) 10 x 26


Theta1_grad= Delta1/m;   
Theta2_grad= Delta2/m;   

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

