function [cost nn_params] = gradientDescent(initial_nn_params,network, X, y, digit, lambda, iter)
%subtract gradient from Theta

	[cost gradient] = nnCostFunction(initial_nn_params, network, X, y, digit, lambda);
	nn_params = initial_nn_params - gradient;
	fprintf('\nIteration %d | Cost %f',iter,cost);

end	