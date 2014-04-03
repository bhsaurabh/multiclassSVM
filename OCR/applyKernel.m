function [ X_recoded ] = applyKernel( inputX, X_train )
% Apply the Gaussian Kernel to input data
% Get new number of features that are used by SVM

% number of training examples = number of landmark points
m = size(X_train, 1);

X_recoded = zeros(1, m);
for i = 1: m
    % last param should be sigma_svm and not simply 0.1
    X_recoded(1, i) = gaussianKernel(inputX, X_train(i, :), 0.1);
end

end

