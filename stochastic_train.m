function [] = stochastic_train
	addpath('function\nn_functions');
	addpath('function\util');
	
	%load data
	
	%this loads Train_X, Train_y
	load('data\general\train.mat');
	%this loads CV_X, CV_y
	load('data\general\cv.mat');
	%this loads Test_X, Test_y
	load('data\general\test.mat');
	
	
	network=[size(Test_X,2); 50; 50; 10];
	num_layers = size(network,1);
	lambda = [0.9; 0.9; 0.9];
	%number of iterations to run a small batch
	iter = 30;

	%stochastic batch size
	samples = 100;

	[Theta cost] = learnStochastic(network, ...
							Train_X, Train_y, 0, lambda, iter,samples);

		
	
	pred = predict(Theta,Train_X);
	train_acc = mean(double(pred == Train_y)) * 100;		
	fprintf('\nTraining Accuracy: %f |\tlambda: %f\n', train_acc, i);	
	
	pred = predict(Theta,CV_X);
	cv_acc = mean(double(pred == CV_y)) * 100;		
	fprintf('\nCV Accuracy: %f |\tlambda: %f\n', cv_acc, i);
	
	pred = predict(Theta,Test_X);
	
	disp('Writing Test Output... \n');
	%writing the headers first
	save_name = sprintf('output\\%s_Theta%s.csv','DigitRec',datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));
	out_id = fopen(save_name,'w+');
	fprintf(out_id,'%s','ImageId,Label');
	fclose(out_id);
	
	out = (1:28000)';
	out = [out pred];
	dlmwrite (save_name, out, '-append','delimiter',',');
	
end