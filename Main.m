function []  = Main(ind_fair, group_fair,lambda, lvl_n, lvl_loc)
% ind_fair;  % if 1 then cost function gets a penalty of individual fairness (refere to Berk et al.)
% group_fair; % if 1 then cost function gets a penalty of group fairness (refere to Berk et al.
% lvl_n; % number of levels of the protected feature
% lvl_loc; % location of the protected feature in the columns of the data
% lambda; % A hyper-parameter corresponding to the penalty function coefficient

%%
%setting up some global parameters related to the fairness
global s_lvl; % number of observations in each level of the protected feature
global M; % Product of the s_lvl which gives the number of paris that are being compared in the penalty function

%% Load Data
%  The first columns contains the features
%  and last column contains the label.
data = load('Data.txt');
rng(1243);
p = .8;
(N, n) = size(data);  % total number of observations
% spiliting data into train-set and test-set by sampling
index = 1:N; 
train_ind = datasample(index,round(N*p))';
tf = false(N,1); % create logical index vector
tf(train_ind) = true;
train = data(tf,:); 
test = data(~tf,:); 

X = train(:,1:n-1); y = train(:,n); % X and y of train-set
X_tes = test(:,1:10); y_tes = test(:,11); % X and y of test-set
[m_tes, n_tes] = size(X_tes); 
X_tes = [ones(m_tes, 1) X_tes]; % add ones for the intercept term
[m, n] = size(X); % Note that definition of n changed here
X = [ones(m, 1) X]; % Add intercept term to x and X_test

%%
% determining the protected feature specifications
lvl_loc = lvl_loc +1;  %1 added to the columns
s_lvl = zeros(lvl_n,1);
M = 1;
for i=1:lvl_n
    temp = size(X(X(:,lvl_loc)==i)); 
    s_lvl(i,1) = temp(1,1);
    M = M * s_lvl(i,1);
end
if M == 0
	fprintf('M is zero');
  return
end

%% ============ Part 1: Compute Cost and Gradient ============
%  In this part, the cost and gradient for logistic regression is computed.

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y, ind_fair, group_fair,lambda, lvl_n, lvl_loc);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

%% ============= Part 2: Optimizing using fminunc  =============
%  In this part, we will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y,ind_fair, group_fair,lambda, lvl_n, lvl_loc)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%% ============== Part 3: Predict and Accuracies ==============
% Evaluating the model performance

% Compute accuracy on our training and test set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

p_tes = predict(theta, X_tes);
fprintf('Test Accuracy: %f\n', mean(double(p_tes == y_tes)) * 100);

%% ============== Part 4: checking the fairness accross the protected feature levels ==============

%fairness in test set
fair_tra = zeros(lvl_n,1);
fair_tes = zeros(lvl_n,1);
for i=1:lvl_n    
    fprintf('Average response for level %d in trainig set is: %f\n', i , mean(p(X(:,lvl_loc)==i,:)));
end
for i=1:lvl_n
    fprintf('Average response for lvl %d in the test set is: %f\n', i ,  mean(p_tes(X_tes(:,lvl_loc)==i,:)));
end
    
end

