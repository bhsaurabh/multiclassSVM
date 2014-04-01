function newX = useKernels(X, sigma)

m = size(X, 1);

% convert x to kernels
% We have m training examples => m landmarks
newX = zeros(m, m);
for i = 1:m
	% calculate gaussian kernel for ith training example
	for j = 1:m
		% ith training example
		fprintf(['\nCalculating kernel on %d, %d'], i, j);
		exampleI = X(i, :);
		landmarkJ = X(j, :);
		fIJ = gaussianKernel(exampleI, landmarkJ, sigma);
		newX(i, j) = fIJ;
	end
end
% append f0 = 1 to every training example
newX = [zeros(m, 1) newX];
fprintf('\nKernel calculations complete');
end