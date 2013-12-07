function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H = X * theta;
sqrErrors = (H - y).^2;
term1 = (1/(2*m))*sum(sqrErrors);
theta_mod = theta([2:size(theta,1)],:); %skip theta_0 (or in octave, theta_1)
term2 = (lambda/(2*m))*sum( theta_mod.^2  );
J = term1 + term2;

% unregularized grad.  We will later just take the fist row (for theta-zero)
grad = (1/m) * ((H-y)'*X)';
regTerm = (lambda/m) * theta;
% the first theta term will remail the same.  Save it first.
grad1 = grad(1);
% add the reg term to grad.  We will discard and replace the first row.
grad = grad + regTerm;
% then we replace the first row (for theta-zero)
grad(1) = grad1;












% =========================================================================

grad = grad(:);

end
