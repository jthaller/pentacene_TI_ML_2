#don't forget scaling and k-fold cross validation
# suggestion to add keras deep learning
#IMPORTS
import numpy as np
import pandas as pd
import pickle
from pprint import pprint as pp
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense

sns.set()
%matplotlib inline

atom_types_df = pd.read_pickle("atom_types.pic")
# print(atom_types)
coulomb_df = pd.read_pickle("coulomb_interactions.pic")
# print(coulomb_df)
TI_df = pd.read_pickle("TI.pic")
# print(TI_df)

coulomb_df = StandardScaler().fit_transform(coulomb_df)

x = coulomb_df
# x_enc = OneHotEncoder(categories='auto').fit_transform(x)
y = np.log10(TI_df)
x.shape
y.shape
# label_enc = LabelEncoder()
# x_enc = label_enc.fit_transform(x)
# y_enc = label_enc.fit_transform(y)
# x_enc = label_enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


#principal component analysis
#
# pca = PCA(n_components='mle',svd_solver='full')
# pca.fit(x_train,y_train)
# print(pca.explained_variance_ratio_)
# x_train_pca = pca.transform(x_train)
# x_test_pca = pca.transform(x_test)
# print(x_train.shape, x_test.shape)

# Linear Regression --------------------
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
linreg = LinearRegression()
linreg.fit(x_train, y_train)
Y_pred = linreg.predict(x_test)
# Compute the rmse from sklearns metrics module imported earlier
rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
print("RMSE: %f" % (rmse))
kfold = KFold(n_splits=10, random_state=7)
results = -1*cross_val_score(linreg, x, y, cv=kfold, scoring='neg_mean_squared_error')
# Note to self:
# The unified scoring API always maximizes the score, so scores which need to be minimized are negated in order for the unified scoring API to work correctly.
# The score that is returned is therefore negated when it is a score that should be minimized and left positive if it is a score that should be maximized.
results
results.mean()
print(f"mean_squard_error: {results.mean()}\nstandard_deviation: {results.std()}")
plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_test,Y_pred)

## XGBoost ----------------------
#booster [default=gbtree] change to gblinear to see. gbtree almost always outperforms though
# xgboost = xgb.XGBClassifier(max_depth=14, n_estimators=1000, learning_rate=0.05,colsample_bytree=1)  #hyperparams
xgboost = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators =10)
xgboost.fit(x_train, y_train)
Y_pred= xgboost.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
print("RMSE: %f" % (rmse))
plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_train,xgboost.predict(x_train))

# Below the crossvalidation was performed on entire dataset
# However better is first perform k-fold cross validation on training data
# After review its performance  on test dataset

#-------k-fold Cross Validation using XGBoost-------
# XGBoost supports the k-fold cross validation with the cv() method
# nfolds is number of cross-validation sets to be build
# More parameters in XGBoost API reference: https://xgboost.readthedocs.io/en/latest/python/python_api.html

#Create Hyper Parameter dictionary params and exclude n_estimators and include num_boost_rounds
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3, 'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}

# nfold is 3, so three round cross validation set using XGBoost cv()
# cv_results include the train and test RMSE metrics for each round
# Separate targetvariable and the rest of the variables

# Convert to optimized data-structure, which XGBoost supports
data_dmatrix = xgb.DMatrix(data=x_train,label=y_train)
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
cv_results.head()

# Print Final boosting round metrics
# The final result may depend upon the technique used, so you may want to try different
# e.g. grid search, random search Bayesian optimization
rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
print("RMSE: %f" % (rmse))
print((cv_results["test-rmse-mean"]).tail(1))



plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_test,Y_pred)
