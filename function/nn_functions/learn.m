function [Theta, cost] = learn(	network, ...
							X, y, digit, lambda, iter)
							
%function takes all the parameters and trains the network							

	addpath('function\nn_functions');
	
	num_layers = size(network,1);
	
	nn_params = [];
	for i = 1: num_layers-1
		parm = randInitializeWeights(network(i),network(i+1));
		nn_params = [ nn_params ; parm(:) ];
	end	
	
	fprintf('\n\nTraining Neural Network for digit %d',digit);
	
	options = optimset('MaxIter', iter);
	[cost, nn_params] = gradientDescent(nn_params, network, X,y,digit,lambda,iter,alpha);
	%for iter = 1:100:size(X,1)-100;

		%costFunction = @(p) nnCostFunction(p, network, X,y, digit, lambda);
		%[nn_params, cost] = fmincg(costFunction, nn_params, options);	
	%	[cost, nn_params] = gradientDescent(nn_params, network, X(i+1:i+100,:),y(i+1:i+100),digit,lambda,iter,alpha);
	%end	
	
	Theta = cell(num_layers-1,1);
	read = 0;
	for i = 1:num_layers - 1
		Theta{i} = reshape(nn_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
						network(i+1), network(i)+1);
						
		read = 	network(i+1) * (network(i) + 1);
	end	
	
end	