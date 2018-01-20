function pr = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
pr = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(size(X,1),1) X]; %gb Size of X is 401 x 1

for i = 1:m
    a2 = sigmoid(Theta1 * X(i,:)'); %gb Size of a2 is 25 x 1
    a2 = [1;a2]; %gb Size of a2 is 26 x 1
    pmat(i,:) = (sigmoid(Theta2 * a2))'; %gb pmat is m by 10. size of sigmoid(Theta2 * a2) is 10 by 1.  
    
end

[YY,pr] = max(pmat,[],2);


    
    
    







% =========================================================================


end
