function [] = digit_rec(digit)
	%pass the digit u want the NN to recognise
	
	addpath('function\nn_functions');
	addpath('function\util');
	
	%initialize paramteres
	load('data\general\train.mat');
	load('data\general\cv.mat');
	network = [1568 ;50; 10];
	num_layers = size(network,1);
	lambda = 1.2;
	accuracy = 0;
	best_lambda = 0;
	best_Theta  = cell(num_layers-1,1);
	iter = 100;
	
	fig = getErrorFigure();
	x = [];
	train = [];
	cv = [];

    %for i = lambda
		%train the NN	
		
		lm = ones(num_layers-1,1) .* i;
		[Theta, cost] = learn( network, Train_X, Train_y, digit, lambda, iter );
		
		%test it against CV
		pred = predict(Theta,CV_X);
		cv_acc =  mean(double(pred == CV_y)) * 100;
		
              
		pred = predict(Theta,Train_X);
		train_acc = mean(double(pred == Train_y)) * 100;
		%x = [ x ; i ];
		train = [ train ; train_acc];
		cv = [ cv ; cv_acc];
		%[x train cv] = plotError(fig,x,train,cv);
		
		if(accuracy < cv_acc)
			best_lambda = lambda;
			best_Theta = Theta;
		end
		
		fprintf('\nCV Accuracy: %f |\tlambda: %f\n', cv_acc, i);
		fprintf('\nTraining Accuracy: %f |\tlambda: %f\n', train_acc, i);
		
		
	%end
	save_name = sprintf('data\\theta\\%s_%d_Theta%s.mat','DigitRec',digit,datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));
	%saving theta
	fprintf('\n\nSaving Theta for lambda %f in %s\n',best_lambda,save_name);
	save(save_name,'best_Theta');
end