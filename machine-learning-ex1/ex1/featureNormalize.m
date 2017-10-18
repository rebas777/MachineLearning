function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));    % size(X,2) means #column of X
sigma = zeros(1, size(X, 2));
mean = 0;
n = size(X, 2);
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
%
%               Note that X is a matrix where each column is a
%               feature and each row is an example. You need
%               to perform the normalization separately for
%               each feature.'
%
% Hint: You might find the 'mean' and 'std' functions useful.
%
for i = 1:n
    Xi = X(:,i);
    sum = 0;
    for(j = 1:size(Xi,i))
        sum  = sum + Xi(j);
    end     %这里调用mean()函数会莫名报错，于是手动实现了mean，但是通不过测试
    mean = sum/size(Xi,i);
    deviation = std(Xi);
    Xi = (Xi-mean)/deviation;
    X_norm(:,i) = Xi;
end







% ============================================================

end
