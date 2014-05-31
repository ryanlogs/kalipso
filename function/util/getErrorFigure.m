function handle = getErrorFigure()
%creates a figure, sets the parameters and returns the handle
	
	fig = figure('Name','Variance/Bias');
	handle = axes('Parent',fig);
	xlabel('Iterations');
	ylabel('Error %');
end	