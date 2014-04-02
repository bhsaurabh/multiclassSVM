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

% =============== Normalize features in data ===========================
[X_norm, mu, sigma] = featureNormalize(X);

% =============== Convert y into form for 1-vs-all =====================
y_new = makeClasses(y, num_labels);

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
    % training X = X_train
    % training Y = y_train(:, i)
    fprintf(['Training SVM for detecting class: %d\n'], i);
    svm_array(i) = svmTrain(X_train, y_train(:, i), C_svm, @(x1, x2) gaussianKernel(x1, x2, sigma_svm));
end


% ============= Predict for all SVMs =================================
% Use cross validation set

   inputX = X_cv(1, :);
   correctY = y_cv(1, :)
   
   pred = zeros(num_labels);
   % predict with SVMs
   for j = 1:length(svm_array)
       pred(j) = svmPredict(svm_array(j), inputX)
   end

