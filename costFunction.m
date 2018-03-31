function [J, grad] = costFunction(theta, X, y)
%   COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

%% setting up some parameters related to the fairness
global ind_fair;  % if 1 then cost function gets a penalty of individual fairness (refere to Berk et al.)
global group_fair; % if 1 then cost function gets a penalty of group fairness (refere to Berk et al.
global lvl_n; % number of levels of the protected feature
global lvl_loc; % location of the protected feature in the columns of the data +1
global s_lvl; % number of observations in each level of the protected feature
global M; % Product of the s_lvl which gives the number of paris that are being compared in the penalty function
global lambda; % A hyper-parameter corresponding to the penalty function coefficient 

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));
h = sigmoid(X * theta);

if ind_fair == 0 && group_fair == 0
    J = -(1 / m) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) ) ; % regular logistic regression loss function
end

%% 
% Individual Fairness penalty
if ind_fair == 1
    p_x = 0;
    for i=1:lvl_n
        for j=i+1:lvl_n     
            wx_2 = (repmat(X(X(:,lvl_loc)==i,:)*theta,1,s_lvl(j)) - ...
                repmat((X(X(:,lvl_loc)==j,:)*theta)',s_lvl(i),1)).^2;
            
            d_fun = penalty( repmat(y(X(:,lvl_loc)==i,:),1,s_lvl(j)) - ...
                repmat(y(X(:,lvl_loc)==j,:)',s_lvl(i),1));
            p_x = p_x + sum(sum(d_fun.* wx_2)); 
        end
    end
    J = -(1 / m) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) ) + 1/M *lambda* p_x;
end

%%
% Group fairness penalty
if group_fair == 1
    p_x = 0;
    for i=1:lvl_n
        for j=i+1:lvl_n     
            wx_2 = (repmat(X(X(:,lvl_loc)==i,:)*theta,1,s_lvl(j)) - ...
                repmat((X(X(:,lvl_loc)==j,:)*theta)',s_lvl(i),1));
            
            d_fun = penalty( repmat(y(X(:,lvl_loc)==i,:),1,s_lvl(j)) - ...
                repmat(y(X(:,lvl_loc)==j,:)',s_lvl(i),1));
            p_x = p_x + sum(sum(d_fun.* wx_2)); 
        end
    end
    J = -(1 / m) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) ) + 1/M * lambda * p_x^2;
end
    

for i = 1 : size(theta, 1)
    grad(i) = (1 / m) * sum( (h - y) .* X(:, i) );
end

end
