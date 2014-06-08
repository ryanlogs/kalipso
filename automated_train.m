Main_Theta = cell(10,1);

for digit = 0:9
	fprintf('Training digit %d\n',digit);
	lm = digit_rec(digit);
	Main_Theta{digit+1} = alternate_train(digit,lm);
end	

save_name = sprintf('data\\theta\\%s_%s_Theta%s.mat','DigitRec','Main_Theta',datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));
	%saving theta
	%fig = getErrorFigure();
	%plotError(fig,x,train,cv);
fprintf('\n\nSaving Theta for lambda %f in %s\n',lm,save_name);
save(save_name,'Main_Theta');

%use all theta to predict final accuracy

%this loads Train_X, Train_y
load('data\general\train.mat');
%this loads CV_X, CV_y
load('data\general\cv.mat');
%this loads Test_X, Test_y
load('data\general\test.mat');

addpath('function\nn_functions');

pred = main_predict(Main_Theta,Train_X);
train_acc = mean(double(pred == Train_Y)) * 100;		
fprintf('\nTraining Accuracy: %f\n', train_acc);	
	
pred = main_predict(Main_Theta,CV_X);
cv_acc = mean(double(pred == CV_Y)) * 100;		
fprintf('\nCross Validation Accuracy: %f\n', cv_acc);	
		
pred = main_predict(Main_Theta,Test_X);
		

disp('Writing Test Output... \n');
%writing the headers first
save_name = sprintf('output\\%s_Theta%s.csv','DigitRec',datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));
out_id = fopen(save_name,'w+');
fprintf(out_id,'%s','ImageId,Label');
fclose(out_id);

out = (1:28000)';
out = [out pred];
dlmwrite (save_name, out, '-append','delimiter',',');
fprintf('saving Theta to : %s\n', save_name);

fprintf('Done!!!\n');