function [J grad] = kaput_nnThetaCostFunction(nn_params, ...
                                   network, ...
                                   X, y, digit, lambda)

	
	addpath('function\nn_functions');
	addpath('function\util');
	m = size(X,1);	
	
	num_layers = length(network);
	num_lables = network(num_layers);
	Theta = cell(num_layers-1,1);
	read = 0;

	for i = 1:num_layers - 1
		Theta{i} = reshape(nn_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
						network(i+1), network(i)+1);
						
		read = 	read + network(i+1) * (network(i) + 1);
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
			Z{i} = A{i-1} * (Theta{i-1})';
			A{i} = hyperbolic(Z{i});
			if(i~=num_layers)
				A{i} = [ ones(m,1) , A{i} ];
			end		
		end	
	end
	
	%setting output vector	
	%Y = [-1.*y , y];
	Y = -1.*ones(m, num_lables);
	for i = 1:m,
		Y(i,y(i)+1) = 1;
	end;

	% mm = max(A{num_layers})
	%P = ((A{num_layers})./ 1.7159 + 1)./2;
		
	%P = errorFunction(A{num_layers},Y);
	
	% mm = max(P(:))
	% ss = min(P(:))
	
	%Q = (Y./1.7159 + 1)./2; 
	%p1 = Q.* log(P);
	%p2 = (1 - Q) .* log(1 - P);
	
	%J = sum(p1 + p2);
	J = sum(sum(errorFunction(A{num_layers},Y))) / (m);

	reg = 0;
	for i = 1:num_layers-1
		t = Theta{i}(:,2:end);
		reg = reg + sum(t(:).^2) * lambda(i);
	end
	
	reg = reg / (2*m);
	
	J = J + reg;

	%computing gradient
	
	delta = cell(num_layers,1);
	del = cell(num_layers,1);
	
	for i = num_layers:-1:1
		if(i==num_layers)
			%errGradient = Y.*(-2/3).*((A{i}./2 .* hyperbolicGradient(Z{i})).*(Y-A{i}./2) + (1.7159 - A{i}.^2./4).*(hyperbolicGradient(Z{i})./2));
			del{i} = (A{i} - Y).*hyperbolicGradient(Z{i});
			%del{i} = errGradient;
		else 
			if(i == 1)
				delta{i} = (del{i+1})' * A{i};  
			else
				del{i} = del{i+1} * Theta{i};
				del{i} = del{i}(:,2:end) .* hyperbolicGradient(Z{i});
				delta{i} = (del{i+1})' * A{i};  	
			end
		end
	end
	
	Theta_grad = cell(num_layers-1,1);
	for i = 1:num_layers - 1
		Theta_grad{i} = delta{i} ./ m;
	end

	grad = [];
	for i = 1:num_layers - 1
		Theta_grad{i}(:,2:end) = Theta_grad{i}(:,2:end)  + Theta{i}(:,2:end) .* (lambda(i)/m);
		
		% Unroll gradients	
		grad = [grad ; Theta_grad{i}(:)];
	end
	
end
