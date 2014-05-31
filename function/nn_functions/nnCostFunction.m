function [J grad] = nnCostFunction(nn_params, ...
                                   network, ...
                                   X, y, digit, lambda)

	
	m = size(X,1);	
	
	num_layers = length(network);
	Theta = cell(num_lables-1,1);
	read = 0;
	for i = 1:num_lables - 1
		Theta{1} = reshape(nn_params(read + 1:network(i+1) * (network(i) + 1)), ...
						network(i+1), network(i)+1);
						
		read = 	network(i+1) * (network(i) + 1);
	end		
	
% You need to return the following variables correctly 
	J = 0;
		
% computing cost
	A = cell(num_layers,1);
	Z = cell(num_layers,1);
	for i=1:num_layers
		if(i==1)
			A{i} = [ones(m,1) , X];
		else
			Z{i} = A{i-1} * Theta{i-1};
			A{i} = sigmoid(Z{i});
			if(i!=num_layers)
				A{i} = [ ones(m,1) , A{i} ];
			end		
		end		
	end
	
	%setting output vector	
	Y = zeros(m,1);
	for i = 1:m
		if(y(i)==digit)
			Y(i,1) = 1;
		end	
	end
	
	p1 = Y .* log(A{num_layers});
	p2 = (1 - Y) .* log(1 - A{num_layers});
	
	J = sum(p1 + p2) ;
	J = sum(J) / (-1 * m);

	reg = 0;
	for i = 1:num_lables-1
		t = Theta{i}(:,2:end);
		reg = sum(t(:).^2) * lambda(i);
	end
	
	reg = reg / (2*m);
	
	J = J + reg;
	
	%computing gradient
	
	delta = cell(num_layers,1);
	del = cell(num_layers,1);
	
	for i = num_layers:-1:1
		if(i==num_layers)
			del{i} = 2 .* (A{i} - Y) .* sigmoidGradient(A{i});
		else
			del{i} = delta{i+1} * Theta{i};
			del{i} = del{i}(:,2:end) .* sigmoidGradient(Z{i});
			delta{i} = (del{i+1})' * A{i};  
		end
	end
	
	Theta_grad = cell(num_lables-1,1);
	for i = 1:num_lables - 1
		Theta_grad{i} = delta{i} ./ m;
	end

	grad = []
	for i = 1:num_lables - 1
		Theta_grad{i}(:,2:end) = Theta_grad{i}(:,2:end)  + Theta{i}(:,2:end) .* (lambda(i)/m);
		
		% Unroll gradients	
		grad = [grad ; Theta_grad{i}(:)];
	end
	
	
end
