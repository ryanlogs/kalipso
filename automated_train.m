for digit = 0:9
	fprintf('Training digit %d\n',digit);
	lm = digit_rec(digit);
	alternate_train(digit,lm);
end	
	