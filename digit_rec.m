function [best_lambda] = digit_rec(digit)
	%pass the digit u want the NN to recognise
	
	addpath('function\nn_functions');
	addpath('function\util');
	
	%initialize paramteres
	load('data\general\train.mat');
	load('data\general\cv.mat');
	network = [size(Train_X,2) ;50;  2];
	num_layers = size(network,1);
	lambda = 0.1:0.1:1.3;
	accuracy = 0;
	best_lambda = 0;
	best_Theta  = cell(num_layers-1,1);
	iter = 80;
	
	x = [];
	train = [];
	cv = [];

	Train_Y = -1.*ones(size(Train_y,1),1);
	for i = 1:size(Train_y,1)
		%Y(i,y(i)+1) = 1;
		if(Train_y(i)==digit)
			Train_Y(i) = 1;
		end
	end;

	CV_Y = -1.*ones(size(CV_y,1),1);	
	for i = 1:size(CV_y,1)
		%Y(i,y(i)+1) = 1;
		if(CV_y(i)==digit)
			CV_Y(i) = 1;
		end	
	end;
	
    for i = lambda
		%train the NN	
		
		lm = ones(num_layers-1,1) .* i;
		[Theta, cost] = learn( network, Train_X, Train_Y, digit, lm, iter );
		
		%test it against CV
		pred = predict(Theta,CV_X);
		cv_acc =  mean(double(pred == CV_Y)) * 100;
             
		pred = predict(Theta,Train_X);
		train_acc = mean(double(pred == Train_Y)) * 100;
		
		x = [ x ; i ];
		train = [ train ; train_acc];
		cv = [ cv ; cv_acc];
		
		if(accuracy < cv_acc)
			best_lambda = lm;
			best_Theta = Theta;
		end
		
		fprintf('\nCV Accuracy: %f |\tlambda: %f\n', cv_acc, i);
		fprintf('\nTraining Accuracy: %f |\tlambda: %f\n', train_acc, i);
		
		
	end
	%save_name = sprintf('data\\theta\\%s_%d_Theta%s.mat','DigitRec',digit,datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));
	%saving theta
	%fig = getErrorFigure();
	%plotError(fig,x,train,cv);
	%fprintf('\n\nSaving Theta for lambda %f in %s\n',best_lambda,save_name);
	%save(save_name,'best_Theta');
	fprintf('Best Lambda = %f\n',best_lambda)
end