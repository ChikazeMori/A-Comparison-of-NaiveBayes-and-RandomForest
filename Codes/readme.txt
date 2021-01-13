== Instructions ==

*** 'test.m' is the only file that you can run since this folder only contains the test set ***

The two best models of Naïve Bayes and Random Forest are contained in this folder as 'NB_best_model.mat' and 'RF_best_model.mat' respectively.

Since the best NB model was applied PCA, we have a different test set for each model.
The test set for the best NB model is 'X_test_pca.csv' and 'y_test_pca.csv'. They are attributes and classes respectively.
The test set for the best RF model is 'X_test.csv' and 'y_test.csv'. They are attributes and classes respectively.
The test set was created in Python.

'test.m' file calculates classification errors of the two best models, and also visualise confusion matrices of each model.

'validation.m' file was used for validation of all the models and visualisation of the results.

All the other files named '***_model_selection_***.m' were used for training models.

== Library Dependencies ==

'MATLAB' version 9.9

'Statistics and Machine Learning Toolbox' version 12.0

== Software Version ==

All the codes were written in MATLAB R2020b.

It is confirmed that MATLAB R2020a on the City's remote lab can run 'test.m' file without any problem.


== References ==

The dataset:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modelling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

Codes:
MathWorks
https://uk.mathworks.com/