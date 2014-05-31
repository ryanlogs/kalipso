function  [x train test] = plotError(handle,x,test,train)
%adds new data over existing plot
% x : vector for x axis
% y : vector for y axis
	hold on;
	plot(handle,x,train,'red',x,test,'green');
	hold off;
	x = [ x(end,end) ];
	test = [ test(end,end) ];
	train = [ train(end,end) ];
	legend('Train Error %','Test Error %');
end	