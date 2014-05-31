function  [x y] = plotError(handle,x,y)
%adds new data over existing plot
% x : vector for x axis
% y : vector for y axis

	hold on;
	plot(handle,x,y,'red');
	hold off;
	x = [ x(end,end) ];
	y = [ y(end,end) ];
	pause; 
end	