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
                 num_labels,(hidden_layer_size + 1));



% Setup some useful variables

m = size(X, 1);
        
 
% You need to return the following variables correctly 

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X=[ones(size(X,1),1) X];
a1=X;
s1=sigmoid(X*Theta1');
z2=X*Theta1';

s1=[ones(size(s1,1),1) s1];
a2=s1;

s2=sigmoid(s1*Theta2');
z3=s1*Theta2';
a3=s2;
y_matrix=eye(num_labels)(y,:);
J=(-y_matrix.*log(s2))-((1-y_matrix).*log(1-s2));
J=J/m;
J=sum(sum(J));
Theta1=Theta1(:,2:end);
Theta2=Theta2(:,2:end);
sum1=sum(sum(Theta1.^2))+sum(sum(Theta2.^2));
J=J+(sum1*lambda)/(2*m)

%fprintf('Theta1_grad');
%size(Theta1_grad)
%fprintf('Theta2_grad');
%size(Theta2_grad)


del3=a3-y_matrix;
del2=del3*Theta2;
del2=del2.*sigmoidGradient(z2);
%fprintf('del3');
%size(del3)
%fprintf('del3');
%size(del3)
%fprintf('del2');
%size(del2)
%fprintf('a2');
%size(a2)
%fprintf('a1');
%size(a1)

dDel2=del3'*a2;
dDel1=del2'*a1;

Theta1_grad=dDel1/m;
Theta2_grad=dDel2/m;
%fprintf('after theta1_grad');
%size(Theta1_grad)
%fprintf('after theta2_grad');
%size(Theta2_grad)


Theta1_temp= (lambda/m)*Theta1;
Theta1_temp=[zeros(size(Theta1_temp,1),1) Theta1_temp];

Theta2_temp= (lambda/m)*Theta2;
Theta2_temp=[zeros(size(Theta2_temp,1),1) Theta2_temp];
%size(Theta1_temp)
%size(Theta2_temp)
Theta1_grad=Theta1_grad+Theta1_temp;
Theta2_grad=Theta2_grad+Theta2_temp;


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
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];




end
