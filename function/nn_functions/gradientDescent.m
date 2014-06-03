function [cost nn_params] = gradientDescent(initial_nn_params,network, X, y, digit, lambda, iter, alpha)
%subtract gradient from Theta

	[cost gradient_params] = nnCostFunction(initial_nn_params, network, X, y, digit, lambda);
	nn_params = [];
	read = 0;

	for i = 1:size(network,1) - 1
		initial = reshape(initial_nn_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
						network(i+1), network(i)+1);
		gradient = reshape(gradient_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
						network(i+1), network(i)+1);				
		Theta =  alpha(i) .*(initial - gradient);
					
		nn_params = [nn_params(:) ; Theta(:)];				
		read = 	network(i+1) * (network(i) + 1);
	end		
	
	if(mod(iter,10000)==1)
		fprintf('\nIteration %d | Cost %f',iter,cost);
	end	

end	