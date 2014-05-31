function X = addFeatures(mat)
	m = sqrt(size(mat,2));
	r = 1:m;
	X = mat;
	
	top_half = [];
	bottom_half = [];
	
	left_half = [];
	right_half = [];
	
	%row sum
	for i = r
		feature = sum(mat(:,i +(m*(r-1))),2)/m;
		X = [X feature];
		if( i > m/2)
			bottom_half = [ feature bottom_half ];
		else 
			top_half = [ top_half feature ];
		end
	end

	%column sum
	for i = r
		feature = sum(mat(:,r +(m*(i-1))),2)/m;
		X = [X feature];
		if( i > m/2)
			right_half = [ feature right_half ];
		else 
			left_half = [ left_half feature ];
		end
	end	
	vertical_symmetry = right_half - left_half;
	horizontal_symmetry = top_half - bottom_half;

	X = [ X vertical_symmetry horizontal_symmetry ];
end	

