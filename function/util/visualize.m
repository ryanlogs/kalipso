function display_array = display_data(X,img)
% Displays multiple 28x 28 digit images

% X : each line contains 784 pixels of the image
	
	m = size(X,1);
	grid_size = ceil(sqrt(m));
	
	display_array = ones(grid_size*img,grid_size*img) * 0 ;
	
	for i = 1:m
 		digit = reshape(X(i,:),img,img);
		digit = digit';
		
		r0 = floor((i-1)/grid_size) * img + 1;
		c0 = mod(i-1,grid_size) * img + 1;
		r1 = r0 + img - 1;
		c1 = c0 + img - 1;

		display_array(r0:r1,c0:c1) = digit;
	end
	
	imshow(display_array,[-100 109]);
end	