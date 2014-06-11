function [X_norm, mu, sigma] = feature_normalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
m = size(X,1);
%  X_norm = bsxfun(@minus, X, mu);
X_norm = X-(ones(m,1)*mu);
sigma = std(X_norm);
% X_norm = bsxfun(@rdivide, X_norm, sigma);

X_norm = X_norm./(ones(m,1)*sigma);

% ============================================================

end
