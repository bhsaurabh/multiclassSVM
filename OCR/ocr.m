% clear screen and variable space
clc; clear all; close all;

% a few parameters to be set
input_layer_size = 400; % 20x20 images give 400 pixels = 400 features
num_labels = 10;    % 10 classes to classify into

% =========== Load and visualise training data ========================
fprintf('Loading and visualising data...\n');
load('ex3data1.mat');
m = size(X, 1); % number of training samples

% Visualise 100 data points randomly
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

fprintf('Program paused. Press ENTER to continue...\n');
pause;

% =============== Shuffle training array =============================
X = X(rand_indices, :);
y = y(rand_indices, :);

% =============== Normalize features in data ===========================
[X_norm, mu, sigma] = featureNormalize(X);

% =============== Convert y into form for 1-vs-all =====================
y_new = makeClasses(y, num_labels);

% ============== Use dimensionality reduction to speed up training =====
[U, S, X_norm] = pca(X_norm);
pause;

% =============== Split data into training, cv and test sets ===========
% make a 60%, 20%, 20% split
split1 = m * 0.6;
split2 = m * 0.8;
% training set
X_train = X_norm(1:split1, :);
y_train = y_new(1:split1, :);
% cross-validation set
X_cv = X_norm(split1+1: split2, :);
y_cv = y_new(split1+1: split2, :);
% test set
X_test = X_norm(split2+1:end, :);
y_test = y_new(split2+1:end, :);

% =============== Initialise an array of svms (structure) =============
% there are num_labels svm's to consider
svm_array(num_labels) = struct('X', [], 'y', [], 'kernelFunction', 'gaussianKernel', 'b', [], 'alphas', [], 'w', []);
C_svm = 1; sigma_svm = 0.1; % additional SVM parameters


% ============== Train SVMs ===========================================
for i = 1: num_labels
    fprintf('\nChoosing parameters and training SVMs. This might take a long time ....\n');
    % training X = X_train
    % training Y = y_train(:, i)
    fprintf('Training SVM for detecting class: %d\n', i);
    % Do uncomment the line below if time is not an issue
    % Performing parameter selection automatically should vastly improve accuracy
    %[C_svm, sigma_svm] = dataset3Params(X_train, y_train(:, i), X_cv, y_cv(:, i));
    svm_array(i) = svmTrain(X_train, y_train(:, i), C_svm, @(x1, x2) gaussianKernel(x1, x2, sigma_svm));
end

fprintf('\nTraining complete... Press ENTER to continue...\n');
save('svmArray.mat', 'svm_array');

% ============= Predict for all SVMs =================================
% Use cross validation set
fprintf('\n===============\nRunning predictions on test set and deriving accuracy');
count = 0;  % number of correctly predicted terms
for i = 1:size(X_test, 1)
    input = X_test(i, :);
    expected = y_test(i, :);
    % convert expected into a single number from the classes array
    expectedY = 0;
    for j = 1:num_labels
       if expected(j) == 1 
           expectedY = j;
           break;
       end
    end
    % check the expectedY SVM's output
    prediction = svmPredict(svm_array(expectedY), input);
    if prediction == 1
        count = count + 1;
    end
end
success = 100 * count/size(X_test, 1);
format long g;
fprintf('\nCross validation accuracy: %d %\n', success);
format short;
fprintf('\nPaused... Press ENTER to continue...\n');
pause;

% ======================== Evaluate using SVM confidence ============
fprintf('\n ====================================================== \n');
fprintf('\n Evaluating SVM accuracy based on confidence metrics \n');
count = 0;  % counts number of correct results

for i = 1:length(X_test)
    inputX = X_test(i,:);
    expected = y_test(i, :);
    % get numeric y
    expectedY = 0;
    for j = 1:num_labels
       if expected(j) == 1 
           expectedY = j;
           break;
       end
    end
    pred = predict(inputX, X_train, svm_array);
    if pred == expectedY
        count = count + 1;
    end
    fprintf('Predicted: %d ... Actual: %d\n', pred, expectedY);
end

fprintf('\n Accuracy is: %d\n', (count/size(X_test, 1)*100));
