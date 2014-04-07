This is an implementation of Multi-Class classification with Support Vector Machines
SVMs are great for binary classification. This implementation makes use of the one-vs-all method.

ocr.m is the Application entry point. It calls other functions to perform the classification task.

Training data: stored in ex3data.mat - This is a collection of 5000 handwritten digits stored as 20x20 images. 
This means that there are 400 features for every training sample.
This training data is split as follows:
    60 % : Used for training the SVMs (randomise data before use, as data is ordered)
    20 % : Used as cross validaition set, to derive SVM parameters C_svm and sigma_svm
    20 % : Used for testing the classifier

Number of classes is 10 - Classification of digits (0 - 9) => there are 10 classes

Since there are 400 features and 5000 training examples we use the Gaussian kernel as opposed to simple Logistic regression or SVMs without any kernel (linear kernel).

After the SVMs are done classifying the result from the most confident SVM is chosen. Confidence is measured as follows:
    confidence = (theta)' * (inputX)
    where theta: set of parameters to train the SVM
    and inputX: column vector of an input sample onto which the kernel function has been applied

Currently I am able to get an accuracy of 79.83% to 82.6% on the test set, which I believe would increase if C_svm and sigma_svm are chosen automatically using dataset3Params.m (the use of this script has been omitted as it takes a good deal of time to execute)
