% For reproductibity
rng('default');

% Load the dataset
X_train_pca = readtable('X_train_pca.csv','VariableNamingRule','preserve');
y_train_pca = readtable('y_train_pca.csv','VariableNamingRule','preserve');

X_train_pca = X_train_pca{:,:};
y_train_pca = y_train_pca{:,:};

% Naive Bayes Hyperparameter Optimization
Mdl_NB_pca = fitcnb(X_train_pca,y_train_pca,'OptimizeHyperparameters',{'DistributionNames','Width'},'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','KFold',10));

% Save the optimized model
save('NB_final_model_pca.mat','Mdl_NB_pca');