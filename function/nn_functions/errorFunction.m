function [y] = errorFunction(output,teacher)
	output = output./2;
	y =  teacher .* (2/3) .* (1.7159 - output.^2) .* (teacher - output);
end 