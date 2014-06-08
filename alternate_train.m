function [] = alternate_train(digit,lm)

addpath('function\nn_functions');
addpath('function\util');

%load data

%this loads Train_X, Train_y
load('data\general\train.mat');
%this loads CV_X, CV_y
load('data\general\cv.mat');
%this loads Test_X, Test_y
load('data\general\test.mat');


network=[size(Test_X,2); 50; 50; 2];

num_layers = size(network,1);
lambda = ones(num_layers-1,1).*lm;
iter = 300;

%setting initial_nn_params
initial_nn_params = [];
for i = 1: num_layers-1
	parm = randInitializeWeights(network(i),network(i+1));
	initial_nn_params = [ initial_nn_params ; parm(:) ];
end

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

options = optimset('MaxIter', iter);

%training NN, the digit value 0 is just a dummy value, not used inside
costFunction = @(p) nnThetaCostFunction(p, network, Train_X, Train_Y, digit, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);	

% nn_params = initial_nn_params;
% for iter = 1:100 
% [cost, nn_params] = gradientDescent(nn_params,network,Train_X,Train_y,0,lambda,iter,[1.2;0.5;0.5]);
% end

	
%unrolling theta	
Theta = cell(num_layers-1,1);
read = 0;
for i = 1:num_layers - 1
	Theta{i} = reshape(nn_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
					network(i+1), network(i)+1);
						
	read = 	read + network(i+1) * (network(i) + 1);
end

pred = predict(Theta,Train_X);
train_acc = mean(double(pred == Train_Y)) * 100;		
fprintf('\nTraining Accuracy: %f |\tlambda: %f\n', train_acc, i);	

pred = predict(Theta,CV_X);
cv_acc = mean(double(pred == CV_Y)) * 100;		
fprintf('\nCV Accuracy: %f |\tlambda: %f\n', cv_acc, i);



	save_name = sprintf('data\\theta\\%s_%d_Theta%s.mat','DigitRec',digit,datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));
	%saving theta
	%fig = getErrorFigure();
	%plotError(fig,x,train,cv);
	fprintf('\n\nSaving Theta for lambda %f in %s\n',lm,save_name);
	save(save_name,'Theta');

%pred = predict(Theta,Test_X);

%disp('Writing Test Output... \n');
%writing the headers first
%save_name = sprintf('output\\%s_Theta%s.csv','DigitRec',datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));
%out_id = fopen(save_name,'w+');
%fprintf(out_id,'%s','ImageId,Label');
%fclose(out_id);

%out = (1:28000)';
%out = [out pred];
%dlmwrite (save_name, out, '-append','delimiter',',');
%fprintf('saving Theta to : %s\n', save_name);
end