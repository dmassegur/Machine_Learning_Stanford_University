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

C = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
sigma = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
error_bsf = 1000000;
for i = 1 : length(C)
  for j = 1 : length(sigma)

    fprintf('Training SVM for regularization parameters: C = %0.2f, sigma = %0.2f...', C(i), sigma(j));
  
    % Training the SVM model for Ci and sigmaj using training set:
    model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
    
    % Evaluating error on validation set:
    y_pred_val = svmPredict(model, Xval);
    
    % comparing prediction with validation values accuracy:
    accuracy = mean(double(y_pred_val == yval));
    error = mean(double(y_pred_val ~= yval));  ## 1 - accuracy    ~= it's the opposite of ==
    
    % if error is better than bsf, update bsf SVM parameters:
    if error < error_bsf
      fprintf('These are better SVM parameters for this dataset!\n');
      C_bsf = C(i);
      sigma_bsf = sigma(j);
      error_bsf = error;
    else
      fprintf('These are worse SVM parameters for this dataset.\n');
    endif
    
  endfor
endfor

fprintf('Optimal SVM parameters found: C = %0.2f, sigma = %0.2f.\n', C_bsf, sigma_bsf);

% Assigning optimal values:
C = C_bsf;
sigma = sigma_bsf;



% =========================================================================

end
