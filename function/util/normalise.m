function norm_x = normalise(x, r_min, r_max)
% function normalises each column of input feature	

	min_vec =  min(x);
	max_vec =  max(x);
	
	indices = [];
	for i = 1:size(x,2)
		if(max_vec(1,i)-min_vec(1,i) == 0)
			ratio(1,i) = 1;
			indices = [indices ; i];
		else	
			ratio(1,i) = (r_max - r_min) ./  (max_vec(1,i) - min_vec(1,i)); 
		end
	end	
	diff  = x - (ones(size(x,1),1)*min_vec); % useful to dodge a loop wherein you would want to subtract a value column-wise.
	norm_x = ones(size(x,1),1)*ratio .* diff + r_min;
	
	norm_x(:,indices) = 0;
end	