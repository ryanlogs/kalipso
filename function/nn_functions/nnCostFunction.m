function [J grad] = nnCostFunction(nn_params, ...
                                   network, ...
                                   X, y, lambda)

	
	m = size(X,1);	
	num_layers = length(network);
	num_lables = network(4);
	
	input_layer_size = network(1);
	hidden_layer1_size = network(2);
	hidden_layer2_size = network(3);
	output_layer_size = network(4);
	
	Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size+1)), ...
						hidden_layer1_size, input_layer_size+1);
	read = 	hidden_layer1_size * (input_layer_size+1);
	
	Theta2 = reshape(nn_params(read + 1: read + hidden_layer2_size * (hidden_layer1_size+1)), ...
						hidden_layer2_size, hidden_layer1_size+1);
	read = read + hidden_layer2_size * (hidden_layer1_size+1);
	
	Theta3 = reshape(nn_params(read + 1 : end ), ...
						output_layer_size, hidden_layer2_size+1);							
	
	
% You need to return the following variables correctly 
	J = 0;
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));	
	Theta3_grad = zeros(size(Theta3));	
	
	
	% computing cost
	a1 = [ones(m,1), X];

	z2 = (a1 * Theta1');
	a2 = sigmoid(z2);	
	a2 = [ones(m,1), a2];

	z3 = (a2 * Theta2');
	a3 = sigmoid(z3);	
	a3 = [ones(m,1), a3];

	z4 = (a3 * Theta3');
	a4 = sigmoid(z4);

	Y = zeros(m, num_lables);
	for i = 1:m,
		Y(i,y(i)+1) = 1;
	end;
	
	p1 = Y .* log(a4);
	p2 = (1 - Y) .* log(1 - a4);
	
	J = sum(p1 + p2) ;
	J = sum(J) / (-1 * m);
	
	T1 = Theta1(:,2:input_layer_size+1);
	T2 = Theta2(:,2:hidden_layer1_size+1);
	T3 = Theta2(:,2:hidden_layer2_size+1);
	reg = sum(T1(:) .^ 2) * lambda(1) + sum(T2(:) .^ 2)* lambda(2) + sum(T3(:) .^ 2)* lambda(3);
	reg = (reg * lambda(1)) / (2*m);
	
	J = J + reg;
	
	%computing gradient
	del4 = a4 - Y;
	
	del3 = (del4 * Theta3) ;
	del3 = del3(:,2:hidden_layer2_size+1);
	del3 = del3 .* sigmoidGradient(z3);
	
	del2 = (del3 * Theta2) ;
	del2 = del2(:,2:hidden_layer1_size+1);
	del2 = del2 .* sigmoidGradient(z2);
	
	delta3 = del4' * a3;  
	delta2 = del3' * a2;  
	delta1 = del2' * a1;
	
	Theta1_grad = delta1 ./ m;
	Theta2_grad = delta2 ./ m;
	Theta3_grad = delta3 ./ m;
	
	Theta1_grad(:,2:input_layer_size+1) = Theta1_grad(:,2:input_layer_size+1) + Theta1(:,2:input_layer_size+1) .* (lambda(1) / m);
	Theta2_grad(:,2:hidden_layer1_size+1) = Theta2_grad(:,2:hidden_layer1_size+1) + Theta2(:,2:hidden_layer1_size+1) .* (lambda(2) / m);
	Theta3_grad(:,2:hidden_layer2_size+1) = Theta3_grad(:,2:hidden_layer2_size+1) + Theta3(:,2:hidden_layer2_size+1) .* (lambda(3) / m);

	% Unroll gradients
	grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];
	
end