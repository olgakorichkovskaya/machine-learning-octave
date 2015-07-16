function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels,  X, y, lambda)
 
%m = the number of training examples
%n =401 the number of training features, including the initial bias unit.
%h =25 the number of units in the hidden layer - NOT including the bias unit
%r = the number of output classifications 10X, y, lambda)
K= num_labels;
h=hidden_layer_size;
n=input_layer_size + 1;
m = size(X, 1);

Theta1 = reshape(nn_params(1:h * (n)), h, (n));
Theta2 = reshape(nn_params((1 + (h * (n))):end), K, (h + 1));
         
% Part 1: Feedforward the neural network and return the cost in the J.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
% a3, a2, z2, and a1 
a1=[ones(m, 1) X];
z2=a1 * Theta1'; % (m x n) ? (n x h) --> (m x h) 5000x401 * 401*25 = 5000 * 25
a2 = sigmoid(z2);% 5000x25
a2=[ones(m, 1) a2];%??
a3 = sigmoid(a2 * Theta2');

y_mat = eye(K);
y1 = y_mat(y,:);
sum=0;
for i=1:m
 for k=1:K
     add1=-y1(i,k)*log(a3(i,k));
     add2=(1-y1(i,k))*log(1-a3(i,k));
     sum=sum+add1-add2;
 end
end

%regularization
sum1=0;
sum2=0;
for j=1:h
    for k=1:(n-1)
     sum1=sum1+Theta1(j,k+1)*Theta1(j,k+1);
    end    
end

for j=1:K
    for k=1:h
     sum2=sum2+Theta2(j,k+1)*Theta2(j,k+1);
    end    
end
JR=sum1+sum2;
J=sum/m +lambda*JR/(2*m);


%===BACK propagation - matrix form
d3=a3-y1;% mxr 5000x10

% d2 is tricky. It uses the (:,2:end) columns of Theta2. d2=d3*Theta2(no bias),
%then element-wise scaled by sigmoid gradient of z2. 
d2=d3*Theta2(:,2:end);% (m x r) ? (r x h) --> (m x h) 5000x25 BUT Thea2 is r x h+1
d2=d2 .* sigmoidGradient(z2);
Delta1=d2'*a1; % (h x m) ? (m x n) --> (h x n)  25x5000 * 5000x401= 25x 401
Delta2=d3'*a2;  % (r x m) ? (m x [h+1]) --> (r x [h+1]) 10 x 26


Theta1_grad= Delta1/m;   
Theta2_grad= Delta2/m;   
%Regularized Neural Networks

theta1Reg=[zeros(h,1), Theta1(:,2:end)];
theta2Reg=[zeros(K,1), Theta2(:,2:end)];
Theta1_grad=Theta1_grad+lambda*theta1Reg/m;
Theta2_grad=Theta2_grad+lambda*theta2Reg/m;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%===BACK propagation - vector form
DELTA1=zeros(h,n);
DELTA2=zeros(K,(h+1));
for t=1:m
x = X(t,:);
yy=y1(t,:);
A1=[1 , x];
Z2=A1*Theta1'; %h =25 1x401 * 401x25 =1x25 
A2=sigmoid(Z2);%1x25

A2=[1 , A2];
Z3=A2 * Theta2';
A3 = sigmoid(Z3);

 D3=zeros(K,1);
 for k=1:K
 D3(k,1)=(A3(k)-yy(k)); % 1x10
 end
 D2=D3' * Theta2(:,2:end) .* sigmoidGradient(Z2); % 1x10 *10x26 .*1x25 = 1x25

 DELTA1=DELTA1+D2'*(A1); %25x1 * 1x401= 25x401 % d2'*a1 (h x m) ? (m x n) --> (h x n) 
 DELTA2=DELTA2+D3*(A2); %d3'*a2 10x1 * 1x26 =10x26  (r x m) ? (m x [h+1]) --> (r x [h+1])
end

Theta1_grad=DELTA1/m+lambda*theta1Reg/m;
Theta2_grad=DELTA2/m+lambda*theta2Reg/m;
% Unroll gradients in vector form
%grad = [Theta1_grad(:) ; Theta2_grad(:)];
end