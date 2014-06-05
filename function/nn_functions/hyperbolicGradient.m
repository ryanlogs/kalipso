function g = hyperbolicGradient(z)

g = zeros(size(z));

sig = sigmoid(z);
g = (2*1.7159/3) .* (1-tanh(z).^2) ;

end