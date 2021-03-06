function [] = generate_pca_data(loadCSV,haveFeatures)
% split the data into Cross Validation and train sets into 20:80  %
% use the haveFeatures option to specific if new features should be added
% use loadCSV if data should load from csv 

	addpath('function\util');
	
	fprintf('Processing train.csv ...');
	if(loadCSV == 1)
		csv = csvread('data\train.csv',1,0);
		fprintf('Total Rows : %d \tTotal Columns : %d\n',size(csv));
	
		% seperating data based on labels
		bucket = cell(10,1);
		for i = 1:size(csv,1)
			if(mod(i,1000)==0)
				fprintf('Iteration %d \n', i);
			end	
			bucket{csv(i,1) + 1} = [ bucket{csv(i,1) + 1} ; csv(i,:) ]; 
		end
		save('data\general\buckets.mat','bucket');	
	else
		load('data\general\buckets.mat');
	end	

	
	CV_set = [];
	Train_set = [];
	
	%add first 400 samples to CV and rest to Train set
	fprintf('\n\nSplitting train.csv into Train and CV set ...');
	for i = 1:10
		shuffle = randperm(size(bucket{i},1));	
		CV_set = [ CV_set ; bucket{i}(shuffle(1:400),:) ];
		Train_set = [ Train_set ; bucket{i}(shuffle(401:end),:) ];
	end

	%shuffle training set, randomization helps with stochastic GD
	shuffle = randperm(size(Train_set,1));	
	Train_set = Train_set(shuffle,:);
	
	fprintf('\n\tTraining set size: %d \n\tCV set size : %d',size(Train_set,1),size(CV_set,1));	
	
	
	%add features code here ..
	
	%normalize Training set
	%features = [Train_set(:,2:end) Train_set(:,2:end).^2 ];
	
	%Train_X = normalise(features);
	Train_X = Train_set(:,2:end);
	if(haveFeatures==1)
		Train_X = addFeatures(Train_X);
		Train_X = [Train_X Train_X.^2];
	end	
	%apply pca
	Train_X = [Train_X Train_X.^2 Train_X.^3];
	[Train_X, dummy] = apply_pca(Train_X,0,-1);	
	Train_X = normalise(Train_X, -1, 1);
	Train_y = Train_set(:,1);
	
	k = size(Train_X,2);
	
	fprintf('\n\nSaving Training set in data\\general\\train.mat ...');
	save('data\general\train.mat','Train_X','Train_y');
	
	%normalize Training set
	%features = [normalise(CV_set(:,2:end)) normalise(CV_set(:,2:end)).^2];
	%CV_X = normalise(features);
	CV_X = CV_set(:,2:end);
	if(haveFeatures==1)
		CV_X = addFeatures(CV_X);
		CV_X = [CV_X CV_X.^2];
	end	
	
	%%apply pca
	CV_X = [CV_X CV_X.^2 CV_X.^3];
	[CV_X, dummy] = apply_pca(CV_X,1,k);
	CV_X = normalise(CV_X, -1,1);
	CV_y = CV_set(:,1);
	
	fprintf('\n\nSaving CV set in data\\general\\cv.mat ...');
	save('data\general\cv.mat','CV_X','CV_y');
	
	%save test set directly to test.mat
	fprintf('\n\nProcessing test.csv ...');
	Test_set = csvread('data\\test.csv',1,0);
	
	fprintf('\nTotal Rows : %d \tTotal Columns : %d',size(Test_set));
	fprintf('\n\tTest set size: %d ',size(Test_set,1));
	
	%add features code here ..
	
	%normalize Training set
	%features = [ Test_set Test_set.^2];
	%Test_X = normalise(features);
	Test_X = Test_set;
	if(haveFeatures==1)
		Test_X = addFeatures(Test_X);
		Test_X = [Test_X Test_X.^2];
	end	
	
	%%apply pca
	Test_X = [Test_X Test_X.^2 Test_X.^2];
	[Test_X, dummy] = apply_pca(Test_X,1,k);
	Test_X = normalise(Test_X,-1,1);
	
	fprintf('\n\nSaving Test set in data\\general\\test.mat');
	save('data\general\test.mat','Test_X');

	
	fprintf('\n\nData Processing Completed!!!\n');
end