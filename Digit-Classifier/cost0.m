function [cost] = cost0(z)
% Return cost when y = 0
if z <= -1
	cost = 0;
else
	cost = 1 - (1/(1 + exp(-z)));	% make this linear
	cost = -log(cost);	
end
end