<<<<<<< HEAD
function [finalPredict] = main_predict(Main_Theta, X)
=======
function p = main_predict(Main_Theta, X)
>>>>>>> ffa283e83d83658e96af5d4f6a0492273b4383b8
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 10);

for digit = 0:9
	H = X;
	Theta = Main_Theta{digit+1};
	for i = 1:size(Theta)
		H = hyperbolic([ones(m,1) H] * (Theta{i})');
	end
	p(:,digit+1) = H(:,2);
end
	
[dummy, finalPredict] = max(p, [], 2);
finalPredict = finalPredict - 1; 

% =========================================================================

end