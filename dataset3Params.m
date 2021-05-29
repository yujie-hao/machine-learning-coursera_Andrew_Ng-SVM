function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
candidateArray = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
result = ones(64, 3); % error, C, sigma
row = 1;
for i = 1:8 % C
    for j = 1:8 % sigma
        model= svmTrain(X, y, candidateArray(i), @(x1, x2) gaussianKernel(x1, x2, candidateArray(j)));
        p = svmPredict(model, Xval);
        result(row, 1) = mean(double(p ~= yval));
        result(row, 2) = candidateArray(i);
        result(row, 3) = candidateArray(j);
        row = row + 1;
    end
end
resultAccuracy = result(:, 1);
[~, index]= min(resultAccuracy);
C = result(index, 2);
sigma = result(index, 3);
% =========================================================================

end
