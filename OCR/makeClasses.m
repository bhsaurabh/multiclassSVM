function y_new = makeClasses(y, num_labels)
    % Return the results where each result is an array showing which class
    % the resukt belongs to
    
    % useful vaiables
    m = size(y,1);  % number of training examples
    
    % template array
    template = 1:num_labels;
    y_new = zeros(m, num_labels);
    
    for j = 1:m
        y_new(j, :) = (template == y(j));
    end
end