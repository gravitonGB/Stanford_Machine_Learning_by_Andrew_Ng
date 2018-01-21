Y = zeros(num_labels,size(X,1));
for i = 1:num_labels
    for k = 1:m
        Y(i,k) = y(k) == i;
    end
end