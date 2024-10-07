# Section 1: Import Library
import warnings
warnings.filterwarnings("ignore")
import sklearn, scikitplot, os, scipy
from sklearn import preprocessing, model_selection, linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from skopt import BayesSearchCV #pip install scikit-optimize

from sklearn.metrics import roc_auc_score

random.seed(42)
np.random.seed(42)
from sklearn.preprocessing import LabelEncoder

# Define the path where the image will be saved
save_dir_image = '/home/annphan/PycharmProjects/EEGSignal-for-AD/image_new'

# Create the directory if it doesn't exist
os.makedirs(save_dir_image, exist_ok=True)

# Section 2: Dataset Preparation
df_HCvMCI = pd.read_csv('/home/annphan/PycharmProjects/EEGSignal-for-AD/data/MCIvsHCFourier.csv')
df_MCIvAD = pd.read_csv('/home/annphan/PycharmProjects/EEGSignal-for-AD/data/MCIvsADFourier.csv')
df_ADvHC = pd.read_csv('/home/annphan/PycharmProjects/EEGSignal-for-AD/data/ADvsHCFourier.csv')

HCvMCI = np.asarray(df_HCvMCI)
MCIvAD = np.asarray(df_MCIvAD)
ADvHC = np.asarray(df_ADvHC)

HCvMCI = HCvMCI[:,1:]
MCIvAD = MCIvAD[:,1:]
ADvHC = ADvHC[:,1:]

## Setting label (Y)
YHvM = HCvMCI[:,304]
YMvA = MCIvAD[:,304]
YAvH = ADvHC[:,304]

## Setting X
HCvMCI = HCvMCI[:,0:304]
MCIvAD = MCIvAD[:,0:304]
ADvHC = ADvHC[:,0:304]

HCvMCI = preprocessing.scale(HCvMCI, axis=0)
MCIvAD = preprocessing.scale(MCIvAD, axis=0)
ADvHC = preprocessing.scale(ADvHC, axis=0)
# freq_bands = scipy.signal.welch(ADvHC, fs=256)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Section 3: Classification HC and MCI (MCI=37 # Control=23)
############### All models for HC and MCI ###############
# 3.0 Set up Train, Val, Test sets for all datasets
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        HCvMCI,y=YHvM
                        )
        )

XTempTrain = HCvMCI[train_inds,]

### Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YHvM[train_inds]
                        )
        )

### Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds

### Create sets of X and Y data using indices  for HC and MCI
XTrainHvM = HCvMCI[TrainInds,]
YTrainHvM = YHvM[TrainInds]
XValHvM = HCvMCI[ValInds,]
YValHvM = YHvM[ValInds]
XTestHvM = HCvMCI[TestInds,]
YTestHvM = YHvM[TestInds]

##########################################################################################
# # 3.1 Running RVC - HC and MCI - DONE (Acc = 75%)
RXTrainHvM = HCvMCI[train_inds,]
RYTrainHvM = YHvM[train_inds]
RXTestHvM = HCvMCI[test_inds,]
RYTestHvM = YHvM[test_inds]

# Import necessary libraries
# from sklearn_rvm import RVC
from sklearn_rvm import EMRVC
RVCMod = EMRVC(kernel = 'linear',
             verbose = True)
#
# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(RYTrainHvM)
print(y_encoded)
RVCMod.fit(RXTrainHvM,y_encoded)

# RVCMod.fit(RXTrainHvM,RYTrainHvM)

def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = np.zeros((NumRVs, NumRVs))

    for r in range(NumRVs):
        d1 = sum(RVs[np.arange(NumRVs) != r])  # Adjust this logic as needed
        SumD[r] = d1 / (NumRVs - 1)
    #
    # SumD = 0
    # for RVNum in range(1,NumRVs):
    #     d1 = RVs[RVNum-1,]
    #     d2 = sum(np.ndarray.flatten(
    #             RVs[np.int8(
    #                     np.setdiff1d(np.linspace(0,NumRVs-1,NumRVs),RVNum))]))
    #     SumD = SumD + (d1/d2)
    # SumD = SumD/NumRVs
    return SumD

RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(RXTestHvM)
RVCPred2 = RVCMod.predict(RXTestHvM)

 # Convert predicted labels back to original if needed
RVC_y_pred_labels = le.inverse_transform(RVCPred2)

# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(RYTestHvM,RVCPred1, title='ROC: HC vs MCI using RVC')
# plt.title("ROC: HC vs MCI using RVC", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_HCandMCI_EMRVC.png'))
plt.show()

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(RYTestHvM,RVC_y_pred_labels)
plt.title("Confusion Matrix: HC vs MCI using RVC", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_HCandMCI_EMRVC.png'))
plt.show()

accuracy_HCandMCI_EMRVC = (RVC_y_pred_labels == RYTestHvM).mean() * 100.
print("RVC classification HC and MCI accuracy : %g%%" % accuracy_HCandMCI_EMRVC)

##########################################################################################
## 3.2 Running RLR - HC and MCI - DONE (Acc = 58.3333%)
## Testing for multicollinearity
# Testing for multicollinearity

coef1 = np.corrcoef(HCvMCI, rowvar=False)
plt.hist(coef1)
plt.title('Histogram of HCvMCI Correlation Coefficients')
plt.show()

ncores = 2
# grid = {
#     'C': np.linspace(1e-10,1e5,num = 100), #Inverse lambda
#     'penalty': ['l1']
# }
# paramGrid = sklearn.model_selection.ParameterGrid(grid)
param_grid = {
    'C': (1e-10, 1e5),  # Inverse lambda
    'penalty': ['l1']
}


# Create the logistic regression model
RLRMod = sklearn.linear_model.LogisticRegression(tol=1e-10, random_state=100, n_jobs=ncores, solver='liblinear')

# Use BayesSearchCV for hyper-parameter optimization
opt = BayesSearchCV(RLRMod, param_grid, n_iter=100, n_jobs=ncores, scoring='roc_auc', cv=5)

# Fit the model
opt.fit(XTrainHvM, YTrainHvM)

# Get the best model and score
bestModel = opt.best_estimator_
bestScore = opt.best_score_
allModels = opt.cv_results_['params']
allScores = opt.cv_results_['mean_test_score']

print(f"Best Model: {bestModel}")
print(f"Best Score (AUC): {bestScore}")

# [bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
# paramGrid = paramGrid,
# X_train = XTrainHvM,
# y_train = YTrainHvM,
# X_val = XValHvM,
# y_val = YValHvM,
# metric = sklearn.metrics.roc_auc_score,
# n_jobs = ncores,
# scoreLabel = 'AUC')
#
# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestHvM)[:, 1]  # Probabilities for the positive class
RLRTestPred2 = bestModel.predict(XTestHvM)

# Output predictions
print("Predicted probabilities:", RLRTestPred)
print("Predicted classes:", RLRTestPred2)

# # Plot Receiver Operating Characteristic (ROC) Curve
# scikitplot.metrics.plot_roc(YTestHvM,RLRTestPred,title = 'LR with LASSO')
# plt.savefig(os.path.join(save_dir_image,'ROC_HCandMCI_RLR.png'))
# plt.show()
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestHvM,RLRTestPred2)
plt.title("Confusion Matrix: HC vs MCI using LR", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_HCandMCI_LR.png'))
plt.show()

accuracy = (RLRTestPred2 == YTestHvM).mean() * 100.
print("RLR classification HC and MCI accuracy : %g%%" % accuracy)

##########################################################################################
# # 3.3 Running  RF - HC and MCI - DONE (Acc = 58.33%)
# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainHvM,YTrainHvM)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = np.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if
               (abs(x) >= np.mean(FeatImp[NZInds[0]])
               + 4*np.std(FeatImp[NZInds[0]]))]
ThreshVal = np.mean(FeatImp[NZInds[0]]) + 2*np.std(FeatImp[NZInds[0]])
Keep_NZInds = np.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]

Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestHvM)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestHvM)

scikitplot.metrics.plot_roc(YTestHvM,Pred2_S2, title = 'ROC: HC vs MCI using RF')
# plt.title("ROC: HC vs MCI using RF", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_HCandMCI_RF.png'))
plt.show()

scikitplot.metrics.plot_confusion_matrix(YTestHvM,Pred1_S2)
plt.title("Confusion Matrix: HC vs MCI using RF", fontsize=14)
plt.savefig(os.path.join(save_dir_image, 'confusion_matrix_HCandMCI_RF.png'))
plt.show()

accuracy = (Pred1_S2 == YTestHvM).mean() * 100.
print("RF classification HC and MCI accuracy : %g%%" % accuracy)

##########################################################################################
## 3.4 Running FDA for HC and MCI - DONE (Acc = 50%)
[XTrainFDHvM,YTrainFDHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainHvM,YTrainHvM)
[XTestFDHvM,YTestFDHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestHvM,YTestHvM)

FDMod = LinearDiscriminantAnalysis(tol = 1e-4,solver = 'svd')
FDFit = FDMod.fit(XTrainFDHvM,YTrainFDHvM)

FIvec_FD_HvM = FDMod.coef_

FDTestPredHvM = FDFit.predict_proba(XTestFDHvM)
FDTestPred2HvM = FDFit.predict(XTestFDHvM)

scikitplot.metrics.plot_roc(YTestFDHvM,FDTestPredHvM, title = "ROC: HC vs MCI using FDA")
# plt.title("ROC: HC vs MCI using FDA", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_HCandMCI_FDA.png'))
plt.show()

scikitplot.metrics.plot_confusion_matrix(YTestFDHvM,FDTestPred2HvM)
plt.title("Confusion Matrix: HC vs MCI using FDA", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_HCandMCI_FDA.png'))
plt.show()

accuracy = (FDTestPred2HvM == YTestFDHvM).mean() * 100.
print("FDA classification HC and MCI accuracy : %g%%" % accuracy)

########################################################################################################################
############ Section 4: All models for MCI and AD ############
# 4.0 Set up Train, Val, Test sets for MCI vs AD
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        MCIvAD,y=YMvA
                        )
        )

XTempTrain = MCIvAD[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YMvA[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds


# Create sets of X and Y data using indices  for MCIvAD

XTrainMvA = MCIvAD[TrainInds,]
YTrainMvA = YMvA[TrainInds]
XValMvA = MCIvAD[ValInds,]
YValMvA = YMvA[ValInds]
XTestMvA = MCIvAD[TestInds,]
YTestMvA = YMvA[TestInds]

#######################################################################
#4.1 Running RVC - MCI and AD - DONE (Acc = 50%)

RXTrainMvA = MCIvAD[train_inds,]
RYTrainMvA = YMvA[train_inds]
RXTestMvA = MCIvAD[test_inds,]
RYTestMvA = YMvA[test_inds]

from sklearn_rvm import EMRVC
RVCMod = EMRVC(kernel = 'linear',
             verbose = True)
# RVCMod.fit(RXTrainMvA,RYTrainMvA)
le = LabelEncoder()
y_encodedAvH = le.fit_transform(RYTrainMvA)
RVCMod.fit(RXTrainMvA,y_encodedAvH)
#
def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(np.ndarray.flatten(
                RVs[np.int8(
                        np.setdiff1d(np.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD
#
#
RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(RXTestMvA)
RVCPred2 = RVCMod.predict(RXTestMvA)


RVC_y_pred_labels = le.inverse_transform(RVCPred2)

# Evaluate Performance (DON'T RELY ON ACCURACY!!!)
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(RYTestMvA,RVCPred1, title = "ROC: AD vs MCI using RVC")
# plt.title("ROC: AD vs MCI using RVC", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_ADandMCI_EMRVC.png'))

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(RYTestMvA,RVC_y_pred_labels)
plt.title("Confusion Matrix: AD vs MCI using RVC", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_ADandMCI_EMRVC.png'))

accuracy = (RVC_y_pred_labels == RYTestMvA).mean() * 100.
print("EMRVC classification AD and MCI accuracy : %g%%" % accuracy)

########################################################################
##4.2 Running RLR - MCI and AD - DONE (Acc = 38.88%)
#Testing for multicollinearity
coef2 = np.corrcoef(MCIvAD, rowvar=False)
plt.hist(coef2)
plt.title('Histogram of MCIvAD Correlation Coefficients')
plt.savefig(os.path.join(save_dir_image,'multicollinearity_MCIvAD.png'))
plt.show()

# Define the model and parameter grid
ncores = 2
# grid = {
#     'C': np.linspace(1e-10,1e5,num = 100), #Inverse lambda
#     'penalty': ['l1']
# }
# paramGrid = sklearn.model_selection.ParameterGrid(grid)

param_grid = {
    'C': (1e-10, 1e5),  # Inverse lambda
    'penalty': ['l1']
}

# Create the logistic regression model
# RLRMod = sklearn.linear_model.LogisticRegression(tol = 1e-10, random_state = 100, n_jobs = ncores, verbose = 1)
RLRMod = sklearn.linear_model.LogisticRegression(tol=1e-10, random_state=100, n_jobs=ncores, solver='liblinear')

# Use BayesSearchCV for hyper-parameter optimization
opt = BayesSearchCV(RLRMod, param_grid, n_iter=100, n_jobs=ncores, scoring='roc_auc', cv=5)

# Fit the model
opt.fit(XTrainMvA, YTrainMvA)

# [bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
# paramGrid = paramGrid,
# X_train = XTrainMvA,
# y_train = YTrainMvA,
# X_val = XValMvA,
# y_val = YValMvA,
# metric = sklearn.metrics.roc_auc_score,
# n_jobs = ncores,
# scoreLabel = 'AUC')
#

# Get the best model and score
bestModel = opt.best_estimator_
bestScore = opt.best_score_
allModels = opt.cv_results_['params']
allScores = opt.cv_results_['mean_test_score']

print(f"Best Model: {bestModel}")
print(f"Best Score (AUC): {bestScore}")

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestMvA)[:, 1]  # Probabilities for the positive class
RLRTestPred2 = bestModel.predict(XTestMvA)

# Output predictions
print("Predicted probabilities:", RLRTestPred)
print("Predicted classes:", RLRTestPred2)

# Plot Receiver Operating Characteristic (ROC) Curve
# scikitplot.metrics.plot_roc(YTestMvA,RLRTestPred)

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestMvA,RLRTestPred2)
plt.title("Confusion Matrix: AD vs MCI using LR", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_ADandMCI_LR.png'))

accuracy = (RLRTestPred2 == YTestMvA).mean() * 100.
print("RLR classification AD and MCI accuracy : %g%%" % accuracy)

# ########################################################################
# #4.3 Running RF - MCI and AD - DONE (Acc = 61.11%) - THE RESULT IS NOT STABLE !!!!

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainMvA,YTrainMvA)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = np.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if
               (abs(x) >= np.mean(FeatImp[NZInds[0]])
               + 4*np.std(FeatImp[NZInds[0]]))]
ThreshVal = np.mean(FeatImp[NZInds[0]]) + 2*np.std(FeatImp[NZInds[0]])
Keep_NZInds = np.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]

Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestMvA)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestMvA)

scikitplot.metrics.plot_roc(YTestMvA,Pred2_S2, title = 'ROC: AD vs MCI using RF')
# plt.title("ROC: AD vs MCI using RF", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_ADandMCI_RF.png'))

scikitplot.metrics.plot_confusion_matrix(YTestMvA,Pred1_S2)
plt.title("Confusion Matrix: AD vs MCI using RF", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_ADandMCI_RF.png'))

accuracy = (Pred1_S2 == YTestMvA).mean() * 100.
print("RF classification AD and MCI accuracy : %g%%" % accuracy)
#
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # from scipy import stats
# # FeatImp_RF_MvA_reshape = np.reshape(FeatImp,[19,16])
# # FeatImp_RF_mean = np.mean(FeatImp_RF_MvA_reshape, axis=0)
# # FeatImp_RF_std = np.std(FeatImp_RF_MvA_reshape, axis=0)
# # conf_int = stats.norm.interval(0.95, loc=FeatImp_RF_mean, scale=FeatImp_RF_std)
# #
# # Freq_values = np.linspace(0,30,16)
# # plt.plot(Freq_values,FeatImp_RF_mean, 'o')
# # plt.title("RF Feature Importance by Channel")
# # plt.xlabel("Frequency (Hz)")
# # plt.ylabel("Mean Feature Importance")
# # plt.savefig(os.path.join(save_dir_image,'TESSTFIG.png'))
# # plt.show()
########################################################################
# #4.4 Running FDA for MCI and AD - DONE (Acc = 59.09%)
# Recover Classes Using Fisher's Linear Discriminant Analysis with SVD
[XTrainFDMvA,YTrainFDMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainMvA,YTrainMvA)
[XTestFDMvA,YTestFDMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestMvA,YTestMvA)

FDMod = LinearDiscriminantAnalysis(tol = 1e-4,solver = 'svd')
FDFit = FDMod.fit(XTrainFDMvA,YTrainFDMvA)

FIvec_FD_MvA = FDMod.coef_

FDTestPredMvA = FDFit.predict_proba(XTestFDMvA)
FDTestPred2MvA = FDFit.predict(XTestFDMvA)
scikitplot.metrics.plot_roc(YTestFDMvA,FDTestPredMvA, title = "ROC: AD vs MCI using FDA")
# plt.title("ROC: AD vs MCI using FDA", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_ADandMCI_FDA.png'))

scikitplot.metrics.plot_confusion_matrix(YTestFDMvA,FDTestPred2MvA)
plt.title("Confusion Matrix: AD vs MCI using FDA", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_ADandMCI_FDA.png'))

accuracy = (FDTestPred2MvA == YTestFDMvA).mean() * 100.
print("FDA classification accuracy : %g%%" % accuracy)
#
# # FIvec_FD_MvA = abs(FIvec_FD_MvA)
# # FIvec_FD_MvA =FIvec_FD_MvA.T
# # FIvec_FD_MvA = FIvec_FD_MvA[:,0]
# #
# # FeatMatrix = np.stack([FeatImp, FIvec_FD_MvA], axis = 1)
# # FeatCorr = np.corrcoef(FeatMatrix.T)
# # np.triu(FeatCorr)
# #
# # plt.plot(FeatImp, FIvec_FD_MvA, 'o')
# #
# # FeatImp_FDA_MvA_reshape = np.reshape(FIvec_FD_MvA,[19,16])
# # FeatImp_FDA_mean = np.mean(FeatImp_FDA_MvA_reshape, axis=0)
# #
# # plt.plot(Freq_values,FeatImp_FDA_mean, 'o')
# # plt.title("FD Feature Importance by Channel")
# # plt.xlabel("Frequency (Hz)")
# # plt.ylabel("Mean Feature Importance")
# #
# # plt.savefig(os.path.join(save_dir_image,'TESSTFIG.png'))
# # plt.show()
#
# # np.savetxt("FeatImp_FDA_MvA.csv", FIvec_FD_MvA)
# # np.savetxt("FeatImp_RF_MvA.csv", FeatImp)

############################################################################################################
## Section 5: All models for AD and HC
## 5.0 Set up Train, Val, Test sets for AD vs HC
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        ADvHC,y=YAvH
                        )
        )

XTempTrain = MCIvAD[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YAvH[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds

# Create sets of X and Y data using indices  for ADvHC
XTrainAvH = ADvHC[TrainInds,]
YTrainAvH = YAvH[TrainInds]
XValAvH = ADvHC[ValInds,]
YValAvH = YAvH[ValInds]
XTestAvH = ADvHC[TestInds,]
YTestAvH = YAvH[TestInds]

####################################################################################
##5.1 Running RVC - AD and HC - DONE (Acc = 75%)

from imblearn.over_sampling import SMOTE, ADASYN
RXTrainAvH = ADvHC[train_inds,]
RYTrainAvH = YAvH[train_inds]
RXTestAvH = ADvHC[test_inds,]
RYTestAvH = YAvH[test_inds]

[XTrainResAvH,YTrainResAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTrainAvH,RYTrainAvH)
[XTestResAvH,YTestResAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTestAvH,RYTestAvH)

from sklearn_rvm import EMRVC
RVCMod = EMRVC(kernel = 'linear',
             verbose = True)
# RVCMod.fit(RXTrainAvH,RYTrainAvH)
le = LabelEncoder()
y_encodedAvH = le.fit_transform(YTrainResAvH)
RVCMod.fit(XTrainResAvH,y_encodedAvH)

def RVMFeatImp(RVs): #75%
    NumRVs = RVs.shape[0]
    SumD = np.zeros((NumRVs, NumRVs))

    for r in range(NumRVs):
        d1 = sum(RVs[np.arange(NumRVs) != r])  # Adjust this logic as needed
        SumD[r] = d1 / (NumRVs - 1)
    #
    # SumD = 0
    # for RVNum in range(1,NumRVs):
    #     d1 = RVs[RVNum-1,]
    #     d2 = sum(np.ndarray.flatten(
    #             RVs[np.int8(
    #                     np.setdiff1d(np.linspace(0,NumRVs-1,NumRVs),RVNum))]))
    #     SumD = SumD + (d1/d2)
    # SumD = SumD/NumRVs
    return SumD

def RVMFeatImp(RVs): #75%
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(np.ndarray.flatten(
                RVs[np.int8(
                        np.setdiff1d(np.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD

RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(XTestResAvH)
RVCPred2 = RVCMod.predict(XTestResAvH)

 # Convert predicted labels back to original if needed
RVC_y_pred_labels = le.inverse_transform(RVCPred2)


# Evaluate Performance
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestResAvH,RVCPred1, title = "ROC: AD vs HC using RVC")
# plt.title("ROC: AD vs HC using RVC", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_ADandHC_EMRVC.png'))

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestResAvH,RVC_y_pred_labels)
plt.title("Confusion Matrix: AD vs HC using RVC", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_ADandHC_EMRVC.png'))

accuracy = (RVC_y_pred_labels == YTestResAvH).mean() * 100.
print("RVC classification for AD and HC accuracy : %g%%" % accuracy)

####################################################################################
##5.2: Running RLR - AD and HC - DONE (Acc = 60%)

#Testing for multicollinearity

coef3 = np.corrcoef(ADvHC, rowvar=False)
plt.hist(coef3)
plt.title('Histogram of ADvHC Correlation Coefficients')
plt.show()

[XTrainRLRAvH,YTrainRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH,YTrainAvH)
[XValRLRAvH,YValRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValAvH,YValAvH)
[XTestRLRAvH,YTestRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH,YTestAvH)

ncores = 2
# [bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
# paramGrid = paramGrid,
# X_train = XTrainRLRAvH,
# y_train = YTrainRLRAvH,
# X_val = XValRLRAvH,
# y_val = YValRLRAvH,
# metric = sklearn.metrics.roc_auc_score,
# n_jobs = ncores,
# scoreLabel = 'AUC')
param_grid = {
    'C': (1e-10, 1e5),  # Inverse lambda
    'penalty': ['l1']
}

# Create the logistic regression model
RLRMod = sklearn.linear_model.LogisticRegression(tol=1e-10, random_state=100, n_jobs=ncores, solver='liblinear')

# Use BayesSearchCV for hyper-parameter optimization
opt = BayesSearchCV(RLRMod, param_grid, n_iter=100, n_jobs=ncores, scoring='roc_auc', cv=5)

# Fit the model
opt.fit(XTrainRLRAvH, YTrainRLRAvH)

# Get the best model and score
bestModel = opt.best_estimator_
bestScore = opt.best_score_
allModels = opt.cv_results_['params']
allScores = opt.cv_results_['mean_test_score']

print(f"Best Model: {bestModel}")
print(f"Best Score (AUC): {bestScore}")

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestRLRAvH)[:, 1]  # Probabilities for the positive class
RLRTestPred2 = bestModel.predict(XTestRLRAvH)

# Output predictions
print("Predicted probabilities:", RLRTestPred)
print("Predicted classes:", RLRTestPred2)

feat_imp_RLR = bestModel.coef_
print(feat_imp_RLR)
#
# # Plot Receiver Operating Characteristic (ROC) Curve
# scikitplot.metrics.plot_roc(YTestRLRAvH,RLRTestPred,title = 'AD vs HC with Ridge')
#plt.savefig(os.path.join(save_dir_image,'ROC_ADandHC_RLR.png'))

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestRLRAvH,RLRTestPred2)
plt.title("Confusion Matrix: AD vs HC using LogisticRegression", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_ADandHC_LR.png'))

accuracy = (RLRTestPred2 == YTestRLRAvH).mean() * 100.
print("RLR classification for AD and HC accuracy : %g%%" % accuracy)

################################################################################################
##5.3 Running RF - AD and HC - DONE (Acc = 70%)

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini')
# RFC = RandomForestClassifier(criterion = 'gini', n_estimators=100, random_state=42) #can use 'entropy' instead

[XTrainRFAvH,YTrainRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH,YTrainAvH)
[XValRFAvH,YValRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValAvH,YValAvH)
[XTestRFAvH,YTestRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH,YTestAvH)

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainRFAvH,YTrainRFAvH)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = np.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if
               (abs(x) >= np.mean(FeatImp[NZInds[0]])
               + 4*np.std(FeatImp[NZInds[0]]))]
ThreshVal = np.mean(FeatImp[NZInds[0]]) + 2*np.std(FeatImp[NZInds[0]])
Keep_NZInds = np.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]

Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestRFAvH)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestRFAvH)

scikitplot.metrics.plot_roc(YTestRFAvH,Pred2_S2, title = "ROC: AD vs HC using RF")
# plt.title("ROC: AD vs HC using RFC Model", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_ADandHC_RF.png'))

scikitplot.metrics.plot_confusion_matrix(YTestRFAvH,Pred1_S2)
plt.title("Confusion Matrix: AD vs HC using RFC", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_ADandHC_RF.png'))

accuracy = (Pred1_S2 == YTestRFAvH).mean() * 100.
print("RF classification for AD and HC accuracy : %g%%" % accuracy)

#################################################################################################
##5.4 Running FDA for AD and HC - DONE (Acc = 70%)
# Recover Classes Using Fisher's Linear Discriminant Analysis with SVD
[XTrainFDAvH,YTrainFDAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH,YTrainAvH)
[XTestFDAvH,YTestFDAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH,YTestAvH)

FDMod = LinearDiscriminantAnalysis(tol = 1e-4,solver = 'svd')
FDFit = FDMod.fit(XTrainFDAvH,YTrainFDAvH)

FIvec_FD_AvH = FDMod.coef_

FDTestPredADvsHC = FDFit.predict_proba(XTestFDAvH)
FDTestPred2ADvsHC = FDFit.predict(XTestFDAvH)


scikitplot.metrics.plot_roc(YTestFDAvH,FDTestPredADvsHC, title = "ROC: AD vs HC using FDA")
# plt.title("ROC: AD vs HC using FDA Model", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'ROC_ADandHC_FDA.png'))

scikitplot.metrics.plot_confusion_matrix(YTestFDAvH,FDTestPred2ADvsHC)
plt.title("Confusion Matrix: AD vs HC using FDA", fontsize=14)
plt.savefig(os.path.join(save_dir_image,'confusion_matrix_ADandHC_FDA.png'))

accuracy = (FDTestPred2ADvsHC == YTestFDAvH).mean() * 100.
print("FDA classification for AD and HC accuracy : %g%%" % accuracy)