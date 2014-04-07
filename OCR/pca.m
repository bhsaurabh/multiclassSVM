function [U, S, Z] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

fprintf('\nReducing dimensionality of problem...\n');

% Useful values
[m, n] = size(X);

U = zeros(n);
S = zeros(n);

% covariance matrix
sigma = X' * X / m;

% Singular value decomposition
[U, S, V] = svd(sigma);

% We try to retain 99% of the variance
% We need partial_sum(Sii) >= 0.99 * sum(Sii)
% S is a square matrix whose only non-zero numbers are on the diagonal
totalSum = sum(sum(S));

% Now try to satisfy variance-retention rule
partialSum = 0;
k = 1;
for i = 1:size(S, 1)
    partialSum = partialSum + S(i, i);
    if partialSum >= 0.99 * totalSum
        k = i;
        break;
    end
end

fprintf('Changed dimensionality from %d to %d\n', n, k);
% Project data onto the k eigenvectors
Z = zeros(size(X, 1), k);
for i = 1: m
    input = X(i, :)';
    for j = 1:k
        projection_k = input' * U(:, j);
        Z(i, j) = projection_k;
    end
end

fprintf('Done... Press ENTER to continue...\n');
% =========================================================================

end
