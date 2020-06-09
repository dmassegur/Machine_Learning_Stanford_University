function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    fprintf('\nIteration = %d.\n', iter);
    %theta = theta - alpha/m*sum(
##    t1 = (theta'*X')'
##    t2 = ((theta'*X')' - y)'
##    t3 = ((theta'*X')' - y)' * X
    grad_step = sum( ((theta'*X')' - y)' * X , 1 );
    theta = theta - alpha/m*grad_step';



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('\nCost computed = %f.\n', J_history(iter));
  
##    if iter>1 && abs(J_history(iter)-J_history(iter-1))<0.0001
##      break
##    endif

end

##figure;
##plot(1:iter,J_history,'-');
##xlabel('Iteration number');
##ylabel('Cost J');
##title('Cost function evolution');
##xlim ([min(1:iter),max(1:iter)]);
##grid;

end
