% For reproductibity
rng('default');

% Load the dataset
X_train_smote = readtable('X_train_smote.csv','VariableNamingRule','preserve');
y_train_smote = readtable('y_train_smote.csv','VariableNamingRule','preserve');

% Naive Bayes Hyperparameter Optimization
Mdl_NB_smote = fitcnb(X_train_smote,y_train_smote,'OptimizeHyperparameters',{'DistributionNames','Width'},'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','KFold',10));

% Save the optimized model
save('NB_final_model_smote.mat','Mdl_NB_smote');