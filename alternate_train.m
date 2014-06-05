

addpath('function\nn_functions');
addpath('function\util');

%load data

%this loads Train_X, Train_y
load('data\general\train.mat');
%this loads CV_X, CV_y
load('data\general\cv.mat');
%this loads Test_X, Test_y
load('data\general\test.mat');


network=[size(Test_X,2);50; 10];
num_layers = size(network,1);
lambda = [ 1.2; 0.6 ];
iter = 3000;

%setting initial_nn_params
initial_nn_params = [];
for i = 1:num_layers-1
	parm = randInitializeWeights(network(i),network(i+1));
	initial_nn_params = [ initial_nn_params ; parm(:) ];
end

options = optimset('MaxIter', iter);

%training NN, the digit value 0 is just a dummy value, not used inside
costFunction = @(p) nnThetaCostFunction(p, network, Train_X, Train_y, 0, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);	
	
%unrolling theta	
Theta = cell(num_layers-1,1);
read = 0;
for i = 1:num_layers - 1
	Theta{i} = reshape(nn_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
					network(i+1), network(i)+1);
						
	read = 	read + network(i+1) * (network(i) + 1);
end

pred = predict(Theta,Train_X);
train_acc = mean(double(pred == Train_y)) * 100;		
fprintf('\nTraining Accuracy: %f |\tlambda: %f\n', train_acc, i);	

pred = predict(Theta,CV_X);
cv_acc = mean(double(pred == CV_y)) * 100;		
fprintf('\nCV Accuracy: %f |\tlambda: %f\n', cv_acc, i);