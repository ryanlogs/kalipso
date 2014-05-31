function norm_x = normalise(x)
% function normalises each column of input feature	

	min_vec =  min(x);
	max_vec =  max(x);
	
	r_min =  -1;
	r_max =  1;
	
	ratio = (r_max - r_min) ./  (max_vec - min_vec);
	diff  = x - (ones(size(x,1),1)*min_vec);
	norm_x = ones(size(x,1),1)*ratio .* diff + r_min;
end	