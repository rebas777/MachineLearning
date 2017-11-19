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

% Setup some useful variables
m = size(X, 1);
K = num_labels;
L = 3; 
         
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
%

function output = cost(h0xi, yi)
	output = 0;
	K = size(h0xi, 1);
	for k = 1:K
        output += yi(k)*log(h0xi(k)) + (1-yi(k))*log(1-h0xi(k));
	end
end

function output = cookY(yi_raw, K)
    output = zeros(K,1);
    output(yi_raw) = 1;
end

for i = 1:m
	yi_raw = y(i);
    Xi = X(i,:);
    a1 = [1; Xi'];  
    z2 = Theta1*a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2*a2;
    h0xi = sigmoid(z3);
    yi = cookY(yi_raw, K);  %transform yi_raw from a sigle number to a 0/1 vector 
    J += cost(h0xi, yi);
end

J = (-1)*J/m;

%now add regularization part

reg = 0;

for j = 1:hidden_layer_size
	for k = 2:input_layer_size+1
        reg += Theta1(j,k)^2;
	end
end 

for j = 1:num_labels
	for k = 2:hidden_layer_size+1
		reg += Theta2(j,k)^2;
	end
end

reg = reg * lambda /(2*m);

J = J + reg;




% grads------------------------------------------------------------

DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));
for t = 1:m
    %step 1
    Xi = X(t,:);
    a1 = [1; Xi'];  
    z2 = Theta1*a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    yt_raw = y(t);
    yt = cookY(yt_raw, K);  %transform yi_raw from a sigle number to a 0/1 vector

    %step 2
    delta3 = zeros(num_labels,1);
    delta3 = a3 - yt;

    %step 3
    %delta2 = (Theta2'*delta3).*[1;sigmoidGradient(z2)];  
    delta2 = (Theta2'*delta3).*sigmoidGradient([1;z2]);
    %problem: different demension. Should i put bias unit in or out of the sigmoidGradient function?


    %step 4
    DELTA1 += delta2(2:end)*a1';
    DELTA2 += delta3*a2';

end

%step 5
Theta1_grad = DELTA1/m;
Theta2_grad = DELTA2/m;

%Regularization
for i = 1:hidden_layer_size
    for j = 2:(input_layer_size + 1)
        Theta1_grad(i,j) = Theta1_grad(i,j) + lambda*(Theta1(i,j))/m;
    end
end

for i = 1:num_labels
	for j = 2:(hidden_layer_size + 1)
		Theta2_grad(i,j) = Theta2_grad(i,j) + lambda*(Theta2(i,j))/m;
	end
end


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
