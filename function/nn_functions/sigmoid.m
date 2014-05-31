function g = sigmoid(z)
%Symmetric sigmoid function

	g = 1.7159 .* tanh((2.*z)./3) + 0.01 .*z;
end
