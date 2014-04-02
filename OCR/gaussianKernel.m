function sim = gaussianKernel(x1, x2, sigma)
    % Returns a gaussian kernel between x1 and x2
    
    % Ensure that x1 and x2 are column vectors
    x1 = x1(:); x2 = x2(:);

    % get difference vector
    diff = x1 - x2;
    sim = sum(diff .^ 2);
    sim = sim / (-2 * (sigma ^ 2));
    % apply the exponential
    sim = exp(sim);
end