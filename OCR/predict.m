function [ prediction ] = predict( input, X_train, svm_array )
% Checks the confidence of every SVM in the array
% Outputs the most confident SVM as a prediction

% Step 1: Convert input into format of kernels
input_recoded = applyKernel(input, X_train);

% Step 2: Initialise an array that will store confidence of each SVM
confidence = zeros(1, length(svm_array));

% Step 3: Get confidence of each SVM
for i = 1:length(svm_array)
    svm = svm_array(i);
    % get confidence (convert input to column vector)
    confidence(i) = svm.alphas' * input_recoded';
end

% Step 4: Find maximum confidence and output that as a prediction
max = 1;    % index of max
for i = 1:length(confidence)
    if confidence(i) > confidence(max)
        max = i;
    end
end

% Return the predicted value
prediction = max;
end

