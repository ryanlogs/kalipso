function [] = digit_rec(digit)
	%pass the digit u want the NN to recognise
	
	addpath('function\nn_functions');
	addpath('function\util');
	
	%initialize paramteres
	load('data\general\train.mat');
	load('data\general\cv.mat');
	network = [784 ;100; 100; 2];
	num_layers = size(network,1);
	lambda = 0.1;
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
		[Theta, cost] = learn( network, Train_X, Train_y, digit, [1.2 ;0.6; 0.6], iter );
		
		
		Train_Y = zeros(size(Train_y));
		for j = 1:size(Train_y,1)
			if(Train_y(j)==digit)
				Train_Y(j,1) = 1;
			end	
		end
		
		Train_Y = [ mod(Train_Y+1,2) Train_Y];
		
		CV_Y = zeros(size(CV_y));
		for j = 1:size(CV_y,1)
			if(CV_y(j)==digit)
				CV_Y(j,1) = 1;
			end	
		end
		
		CV_Y = [ mod(CV_Y+1,2) CV_Y ];
		
		%test it against CV
		pred = predict(Theta,CV_X);
		cv_acc =  mean(double(pred == CV_Y(:,2))) * 100;
		
		pred = predict(Theta,Train_X);
		train_acc = mean(double(pred == Train_Y(:,2))) * 100;
		%x = [ x ; i ];
		train = [ train ; train_acc];
		cv = [ cv ; cv_acc];
		%[x train cv] = plotError(fig,x,train,cv);
		
		fprintf('\nCV Accuracy: %f |\tlambda: %f\n', cv_acc, i);
		fprintf('\nTraining Accuracy: %f |\tlambda: %f\n', train_acc, i);
		if(accuracy < cv_acc)
			best_lambda = i;
			best_Theta = Theta;
		end
		
	%end
	save_name = sprintf('data\\theta\\%s_%d_Theta%s.mat','DigitRec',digit,datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));
	%saving theta
	fprintf('\n\nSaving Theta for lambda %f in %s\n',best_lambda,save_name);
	save(save_name,'best_Theta');
end