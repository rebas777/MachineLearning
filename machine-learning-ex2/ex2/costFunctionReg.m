function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% compute J
for i = 1:m
    tmp = -y(i)*log(sigmoid(theta'*X(i,:)'))-(1-y(i))*log(1-sigmoid(theta'*X(i,:)'));
    J += tmp;
end
J = J/m;
tmp_sum = 0;
for j = 2:size(theta)
    tmp_sum += theta(j)^2;
end
J += tmp_sum*lambda/(2*m);

%compute grad
for i = 1:m
    tmp = (sigmoid(theta'*X(i,:)') - y(i))*X(i,1);
    grad(1) += tmp;
end
grad(1) = grad(1)/m;
for j = 2:size(theta)
    for i = 1:m
        tmp = (sigmoid(theta'*X(i,:)') - y(i))*X(i,j);
        grad(j) += tmp;
    end
    grad(j) = grad(j)/m + theta(j)*lambda/m;
end

% =============================================================

end
