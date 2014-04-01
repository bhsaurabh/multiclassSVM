function [X_norm, mu, sigma] = normaliseFeatures(X)
% Normalisation of features is a good idea when using Gaussian SVMs
% Make mean of features 0 and range their values from 0 to 1

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));


% Iterate over multiple features...normalise using multiple training examples
for i = [1: size(X, 2)]
  mu(i) = mean(X(:, i));
  sigma(i) = std(X(:, i));
  for j = [1: size(X, 1)]
    X_norm(j, i) = (X(j, i)-mu(i))/sigma(i);
  end
end

end
