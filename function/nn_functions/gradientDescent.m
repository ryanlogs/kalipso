function [cost nn_params] = gradientDescent(nn_params,network, X, y, digit, lambda, iter)
%subtract gradient from Theta

	[cost gradient] = nnCostFunction(nn_params, network, X, y, digit, lambda);
	nn_params = nn_params - gradient;
	fprintf('Iteration %d | Cost %f',iter,cost);
end	