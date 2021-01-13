% For reproductibity
rng('default');

% Load the optimized models
load('NB_final_model.mat');
load('RF_final_model.mat');
load('NB_final_model_pca.mat');
load('NB_final_model_smote.mat');
load('RF_final_model_smote.mat');

% Load the datasets
X_train = readtable('X_train.csv','VariableNamingRule','preserve');
y_train = readtable('y_train.csv','VariableNamingRule','preserve');
X_test = table2array(readtable('X_test.csv','VariableNamingRule','preserve'));
y_test = table2array(readtable('y_test.csv','VariableNamingRule','preserve'));
X_train_pca = table2array(readtable('X_train_pca.csv'));
y_train_pca = table2array(readtable('y_train_pca.csv'));
X_test_pca = table2array(readtable('X_test_pca.csv'));
y_test_pca = table2array(readtable('y_test_pca.csv'));
X_train_smote = readtable('X_train_smote.csv','VariableNamingRule','preserve');
y_train_smote = readtable('y_train_smote.csv','VariableNamingRule','preserve');

% Extract the best hyperparameters of RF
Result_RF = Mdl_RF.HyperparameterOptimizationResults;
pos = Result_RF.Rank == 1;
BestResultRF = Result_RF(pos,:);
BestNVS = BestResultRF.NumVariablesToSample;
BestNLC = BestResultRF.NumLearningCycles;

% Extract the best hyperparameters of RF_smote
Result_RF_smote = Mdl_RF_smote.HyperparameterOptimizationResults;
pos_smote = Result_RF_smote.Rank == 1;
BestResultRF_smote = Result_RF_smote(pos,:);
BestNVS_smote = BestResultRF_smote.NumVariablesToSample;
BestNLC_smote = BestResultRF_smote.NumLearningCycles;

% Extract the best hyperparameters of NB
Result_NB = Mdl_NB.HyperparameterOptimizationResults;
pos = Result_NB.Rank == 1;
BestResultNB = Result_NB(pos,:);
BestDN = cellstr(BestResultNB.DistributionNames);
BestDN = BestDN{:};
BestW = BestResultNB.Width;

% Extract the best hyperparameters of NB_PCA
Result_NB_pca = Mdl_NB_pca.HyperparameterOptimizationResults;
pos_pca = Result_NB_pca.Rank == 1;
BestResultNB_pca = Result_NB_pca(pos_pca,:);
BestDN_pca = cellstr(BestResultNB_pca.DistributionNames);
BestDN_pca = BestDN_pca{:};
BestW_pca = BestResultNB_pca.Width;

% Extract the best hyperparameters of NB_smote
Result_NB_smote = Mdl_NB_smote.HyperparameterOptimizationResults;
pos_smote = Result_NB_smote.Rank == 1;
BestResultNB_smote = Result_NB_smote(pos,:);
BestDN_smote = cellstr(BestResultNB_smote.DistributionNames);
BestDN_smote = BestDN_smote{:};
BestW_smote = BestResultNB_smote.Width;

% Train a RF model with the best hyperparameters
t = templateTree('Reproducible',true,'NumVariablesToSample',BestNVS);
Mdl_RF_best = fitcensemble(X_train,y_train,'Method','Bag','NumLearningCycles',BestNLC,'Learners',t);
% Cross Validation
Mdl_RF_CV = crossval(Mdl_RF_best);

% Train a RF_smote model with the best hyperparameters
t_smote = templateTree('Reproducible',true,'NumVariablesToSample',BestNVS_smote);
Mdl_RF_best_smote = fitcensemble(X_train_smote,y_train_smote,'Method','Bag','NumLearningCycles',BestNLC_smote,'Learners',t_smote);
% Cross Validation
Mdl_RF_CV_smote = crossval(Mdl_RF_best_smote);

% Train a NB model with the best hyperparameters
Mdl_NB_best = fitcnb(X_train,y_train,'DistributionNames',BestDN,'Width',BestW);
% Cross Validation
Mdl_NB_CV = crossval(Mdl_NB_best);

% Train a NB_PCA model with the best hyperparameters
Mdl_NB_best_pca = fitcnb(X_train_pca,y_train_pca,'DistributionNames',BestDN_pca,'Width',BestW_pca);
% Cross Validation
Mdl_NB_CV_pca = crossval(Mdl_NB_best_pca);

% Train a NB_smote model with the best hyperparameters
Mdl_NB_best_smote = fitcnb(X_train_smote,y_train_smote,'DistributionNames',BestDN_smote);
% Cross Validation
Mdl_NB_CV_smote = crossval(Mdl_NB_best_smote);

% Calculate RF's classification error (Validation)
error_RF_val = kfoldLoss(Mdl_RF_CV);
% Calculate RF_smote's classification error (Validation)
error_RF_val_smote = kfoldLoss(Mdl_RF_CV_smote);
% Calculate NB's classification error (Validation)
error_NB_val = kfoldLoss(Mdl_NB_CV);
% Calculate NB_PCA's classification error (Validation)
error_NB_val_pca = kfoldLoss(Mdl_NB_CV_pca);
% Calculate NB_smote's classification error (Validation)
error_NB_val_smote = kfoldLoss(Mdl_NB_CV_smote);

% Calculate RF's classification error (test)
error_RF_test = loss(Mdl_RF_best,X_test,y_test);
% Calculate RF_smote's classification error (test)
error_RF_test_smote = loss(Mdl_RF_best_smote,X_test,y_test);
% Calculate NB's classification error (test)
error_NB_test = loss(Mdl_NB_best,X_test,y_test);
% Calculate NB_PCA's classification error (test)
error_NB_test_pca = loss(Mdl_NB_best_pca,X_test_pca,y_test_pca);
% Calculate NB_smote's classification error (test)
error_NB_test_smote = loss(Mdl_NB_best_smote,X_test,y_test);

% Visualize errors
error_RF = [BestResultRF.Objective;error_RF_val;error_RF_test];
error_RF_smote = [BestResultRF_smote.Objective;error_RF_val_smote;error_RF_test_smote];
error_NB = [BestResultNB.Objective;error_NB_val;error_NB_test];
error_NB_pca = [BestResultNB_pca.Objective;error_NB_val_pca;error_NB_test_pca];
error_NB_smote = [BestResultNB_smote.Objective;error_NB_val_smote;error_NB_test_smote];
error = {'train';'val';'test'};
error_table = table(error_RF,error_RF_smote,error_NB,error_NB_pca,error_NB_smote,'RowNames',error)

% Visualize confusion matrix of the RF model
[ClassRF, ScoreRF, CostRF] = kfoldPredict(Mdl_RF_CV);
figure
cmRF = confusionchart(y_train{:,:},ClassRF);
cmRF.ColumnSummary = 'column-normalized';
cmRF.RowSummary = 'row-normalized';
cmRF.Title = 'RF Confusion Matrix';

% Visualize confusion matrix of the RF_smote model
[ClassRF_smote, ScoreRF_smote, CostRF_smote] = kfoldPredict(Mdl_RF_CV_smote);
figure
cmRF_smote = confusionchart(y_train_smote{:,:},ClassRF_smote);
cmRF_smote.ColumnSummary = 'column-normalized';
cmRF_smote.RowSummary = 'row-normalized';
cmRF_smote.Title = 'RF SMOTE Confusion Matrix';

% Visualize confusion matrix of the NB model
[ClassNB, ScoreNB, CostNB] = kfoldPredict(Mdl_NB_CV);
figure
cmNB = confusionchart(y_train{:,:},ClassNB);
cmNB.ColumnSummary = 'column-normalized';
cmNB.RowSummary = 'row-normalized';
cmNB.Title = 'NB Confusion Matrix';

% Visualize confusion matrix of the NB_PCA model
[ClassNB_pca, ScoreNB_pca, CostNB_pca] = kfoldPredict(Mdl_NB_CV_pca);
figure
cmNB_pca = confusionchart(y_train_pca,ClassNB_pca);
cmNB_pca.ColumnSummary = 'column-normalized';
cmNB_pca.RowSummary = 'row-normalized';
cmNB_pca.Title = 'NB PCA Confusion Matrix';

% Visualize confusion matrix of the NB_smote model
[ClassNB_smote, ScoreNB_smote, CostNB_smote] = kfoldPredict(Mdl_NB_CV_smote);
figure
cmNB_smote = confusionchart(y_train_smote{:,:},ClassNB_smote);
cmNB_smote.ColumnSummary = 'column-normalized';
cmNB_smote.RowSummary = 'row-normalized';
cmNB_smote.Title = 'NB SMOTE Confusion Matrix';

% Visualise the feature importances
figure
impOOB = oobPermutedPredictorImportance(Mdl_RF_best);
bar(impOOB)
title('Unbiased Predictor Importance Estimates')
ylabel('Importance')
h = gca;
h.XTickLabel = Mdl_RF_best.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% Visualise results of hyperparameter tuning of RF model
x = Result_RF.NumLearningCycles;
z = Result_RF.NumVariablesToSample;
p = Result_RF.Objective;
xv = linspace(min(x), max(x), 100);
zv = linspace(min(z), max(z), 100);
[X,Z] = meshgrid(xv, zv);
P = griddata(x, z, p, X, Z);
figure
s = surfc(X, Z, P,'FaceAlpha',0.5);
xlabel('NumLearningCycles');
ylabel('NumVariablesToSample');
zlabel('Objective');
title('Hyperparameter tuning');

% Save the best model
save('RF_best_model.mat','Mdl_RF_best');
save('NB_best_model.mat','Mdl_NB_best_pca');