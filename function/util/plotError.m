function  [x train cv] = plotError(handle,x,train,cv)
%adds new data over existing plot
% x : vector for x axis
% y : vector for y axis
	hold on;
	plot(handle,x,1-train,'red',x,1-cv,'green');
	hold off;
	x = [ x(end,end) ];
	cv = [ cv(end,end) ];
	train = [ train(end,end) ];
	legend('Train Error %','Cross Validation Error %');
end	