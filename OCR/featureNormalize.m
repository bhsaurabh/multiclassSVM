function [X_norm, mu, sigma] = featureNormalize(X)
    % Normalises all features of X to have mean 0
    % Also makes all features range from 0 to 1
    
    % Initialisations
    X_norm = X;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));
    
    % normalise all features
    for i = 1:size(X, 2)
       mu(i) = mean(X(:, i));
       sigma(i) = std(X(:, i)); % standard deviation
       for j = 1:size(X, 1)
          % Note that j is the row number 
          X_norm(j, i) = (X(j, i) - mu(i)) / sigma(i); 
       end
    end
end