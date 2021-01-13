% Load the dataset
X_train = readtable('X_train.csv','VariableNamingRule','preserve');
y_train = readtable('y_train.csv','VariableNamingRule','preserve');

% Random Forest Hyperparameter Optimization
trees = templateTree('Reproducible',true);
Mdl_RF = fitcensemble(X_train,y_train,'Method','bag','Learners',trees,'OptimizeHyperparameters',{'NumLearningCycle','NumVariablesToSample'},'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','KFold',10));

% Save the optimized model
save('RF_final_model.mat','Mdl_RF');