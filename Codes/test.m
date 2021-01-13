% For reproductibity
rng('default');

% Load the best models
load('NB_best_model.mat');
load('RF_best_model.mat');

% Load the datasets
X_test = table2array(readtable('X_test.csv','PreserveVariableNames',true));
y_test = table2array(readtable('y_test.csv','PreserveVariableNames',true));
X_test_pca = table2array(readtable('X_test_pca.csv'));
y_test_pca = table2array(readtable('y_test_pca.csv'));

% Calculate RF's classification error
error_RF = loss(Mdl_RF_best,X_test,y_test)

% Calculate NB's classification error
error_NB = loss(Mdl_NB_best_pca,X_test_pca,y_test_pca)

% Visualize confusion matrix of the RF model
ClassRF = predict(Mdl_RF_best,X_test);
figure
cmRF = confusionchart(y_test,ClassRF);
cmRF.ColumnSummary = 'column-normalized';
cmRF.RowSummary = 'row-normalized';
cmRF.Title = 'RF Confusion Matrix';

% Visualize confusion matrix of the NB model
ClassNB_pca = predict(Mdl_NB_best_pca,X_test_pca);
figure
cmNB_pca = confusionchart(y_test_pca,ClassNB_pca);
cmNB_pca.ColumnSummary = 'column-normalized';
cmNB_pca.RowSummary = 'row-normalized';
cmNB_pca.Title = 'NB Confusion Matrix';