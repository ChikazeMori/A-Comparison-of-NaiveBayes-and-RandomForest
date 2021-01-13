% Load the dataset
X_train_smote = readtable('X_train_smote.csv','VariableNamingRule','preserve');
y_train_smote = readtable('y_train_smote.csv','VariableNamingRule','preserve');

% Random Forest Hyperparameter Optimization
trees = templateTree('Reproducible',true);
Mdl_RF_smote = fitcensemble(X_train_smote,y_train_smote,'Method','bag','Learners',trees,'OptimizeHyperparameters',{'NumLearningCycle','NumVariablesToSample'},'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','KFold',10));

% Save the optimized model
save('RF_final_model_smote.mat','Mdl_RF_smote');