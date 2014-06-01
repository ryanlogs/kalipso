function g = sigmoid(z)
%Symmetric sigmoid function
	g = 1.0 ./ (1.0 + exp(-z));
end
