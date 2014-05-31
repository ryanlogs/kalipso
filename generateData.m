function [] = generateData(loadCSV,addFeatures)
% split the data into Cross Validation and train sets into 20:80  %
% use the addFeatures option to specific if new features should be added
% use loadCSV if data should load from csv 

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
	
	fprintf('\n\nSaving Training set in data\\general\\train.mat ...');
	save('data\general\train.mat','Train_set');
	
	fprintf('\n\nSaving CV set in data\\general\\cv.mat ...');
	save('data\general\cv.mat','CV_set');
	
	%save test set directly to test.mat
	disp(sprintf('\n\nProcessing test.csv ...'));
	Test_set = csvread('data\\test.csv',1,0);
	
	fprintf('\nTotal Rows : %d \tTotal Columns : %d',size(Test_set));
	fprintf('\n\tTest set size: %d ',size(Test_set,1));
	
	fprintf('\n\nSaving CV set in data\\general\\test.mat');
	save('data\general\test.mat','Test_set');

	
	fprintf('\n\nData Processing Completed!!!\n');
end