function p = predict(Theta, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
H = X;

Theta
for th = 1:size(Theta)
	H = sigmoid([ones(m,1) H] * (Theta{th})');
end

p = H
% =========================================================================


end
