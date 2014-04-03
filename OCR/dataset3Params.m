function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

range_array = [0.01 0.03 0.1 0.3 1, 3, 10 30];
best_prediction = zeros(3, 1);

for C_test = range_array
   for sigma_test = range_array
        fprintf(['Training for C : %d and sigma : %d \n'], C_test, sigma_test);
        % train the SVM
        %size(X)
        %size(y)
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));    
        predictions = svmPredict(model, Xval);
        prediction_accuracy = mean(double(predictions == yval));
        if prediction_accuracy > best_prediction(1)
           fprintf(['==> Best fit C (%d) and sigma (%d) found\n'], C_test, sigma_test);
           best_prediction(1) = prediction_accuracy;
           best_prediction(2) = C_test;
           best_prediction(3) = sigma_test;
        end
   end
end

C = best_prediction(2);
sigma = best_prediction(3);

% =========================================================================

end
