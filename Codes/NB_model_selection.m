% For reproductibity
rng('default');

% Load the dataset
X_train = readtable('X_train.csv','VariableNamingRule','preserve');
y_train = readtable('y_train.csv','VariableNamingRule','preserve');

% Naive Bayes Hyperparameter Optimization
Mdl_NB = fitcnb(X_train,y_train,'OptimizeHyperparameters',{'DistributionNames','Width'},'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','KFold',10));

% Save the optimized model
save('NB_final_model.mat','Mdl_NB');
