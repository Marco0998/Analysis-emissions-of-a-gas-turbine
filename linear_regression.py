# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:04:50 2023

@author: marco
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,median_absolute_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import time

#%%
gt_files=['gt_2011.csv','gt_2012.csv','gt_2013.csv','gt_2014.csv','gt_2015.csv']

columns_extract=['CO','NOX']

gt_completo=[pd.read_csv(file,delimiter=';') for file in gt_files]

gt_completo=pd.concat(gt_completo, axis=0, ignore_index=True)

gt_completo['CDP']=gt_completo['CDP']*1000

gt_completo['GTEP']=gt_completo['GTEP']*100

Y_target = [pd.read_csv(file,delimiter=';',usecols=columns_extract) for file in gt_files]

Y_target=pd.concat(Y_target, axis=0, ignore_index=True)



columns_extract2=['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']



X=[pd.read_csv(file,delimiter=';',usecols=columns_extract2) for file in gt_files]

X=pd.concat(X, axis=0, ignore_index=True) #vettore delle X



#%%


X_train,X_test,y_train,y_test = train_test_split(X,Y_target,test_size=0.2,random_state=0,shuffle=True)




#%%
pca_NOX_lr=PCA(n_components=8)

pipe_lr_NOX=Pipeline([('robust_scaler_NOX',RobustScaler()),('pca_NOX_lr',pca_NOX_lr),('lr_NOX',LinearRegression())])





folds=KFold(n_splits=5)






pipe_lr_NOX.fit(X_train,y_train['NOX'])

cv_scores = cross_val_score(pipe_lr_NOX, X_train, y_train['NOX'], cv=folds,n_jobs=-1, scoring='neg_mean_absolute_error')


y_pred_NOX= pipe_lr_NOX.predict(X_test)

mae_gbr_NOX= mean_absolute_error(y_test['NOX'].values.astype(float), y_pred_NOX)

mape_gbr_NOX=mean_absolute_percentage_error(y_test['NOX'].values.astype(float), y_pred_NOX)*100

mse_gbr_NOX=mean_squared_error(y_test['NOX'].values.astype(float), y_pred_NOX)

made_NOX=median_absolute_error(y_test['NOX'].values.astype(float), y_pred_NOX)

rq_NOX=r2_score(y_test['NOX'].values.astype(float), y_pred_NOX)


#%%
start_time = time.time()


pca_CO_lr=PCA(n_components=5)

pipe_lr_CO=Pipeline([('pca_CO_lr',pca_CO_lr),('lr_CO',LinearRegression())]) #risultato ottimale senza scaler


pipe_lr_CO.fit(X_train,y_train['CO'])

y_pred_CO_lr= pipe_lr_CO.predict(X_test)

mae_gbr_CO_lr= mean_absolute_error(y_test['CO'].values.astype(float), y_pred_CO_lr)

mape_gbr_CO_lr=mean_absolute_percentage_error(y_test['CO'].values.astype(float), y_pred_CO_lr)*100

mse_gbr_CO_lr=mean_squared_error(y_test['CO'].values.astype(float), y_pred_CO_lr)

made_CO=median_absolute_error(y_test['CO'].values.astype(float), y_pred_CO_lr)

rq_CO=r2_score(y_test['CO'].values.astype(float), y_pred_CO_lr)



end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

#%%
pca_CO=PCA(n_components=8)
pipe_lr_l_CO=Pipeline([('pca_CO',pca_CO),('lasso_CO',linear_model.Lasso(alpha=0.2))])
# paramaters_l={'lasso_CO__alpha': [0.1,0.2,0.3,0.4,0.5],'pca_CO__n_components':[2,3,4,5,6,7,8]}
# pipe_search_l= RandomizedSearchCV(
#     estimator=pipe_lr_l_CO,
#     param_distributions=paramaters_l,
#     scoring='neg_mean_squared_error',  # Specify your scoring metric
#     n_iter=4,  # Number of parameter settings that are sampled
#     random_state=42,  # Set a random seed for reproducibility
#     n_jobs=-1  # Use all available CPU cores
# )
pipe_search_l=pipe_lr_l_CO
 
pipe_search_l.fit(X_train,y_train['CO'])



y_pred_CO_l=pipe_search_l.predict(X_test)

mae_gbr_CO_l= mean_absolute_error(y_test['CO'].values.astype(float), y_pred_CO_l)

mape_gbr_CO_l=mean_absolute_percentage_error(y_test['CO'].values.astype(float), y_pred_CO_l)*100

mse_gbr_CO_l=mean_squared_error(y_test['CO'].values.astype(float), y_pred_CO_l)



#%%

pca_NOX_l=PCA()


pipe_lasso_NOX=Pipeline([('robust_scaler_NOX',RobustScaler()),('pca_NOX_l',pca_NOX_l),('lasso_NOX',linear_model.Lasso())])

paramaters_l_NOX={'lasso_NOX__alpha': [0.1,0.2,0.3,0.4,0.5],'pca_NOX_l__n_components':[2,3,4,5,6,7,8]}
pipe_search_l_NOX= RandomizedSearchCV(
    estimator=pipe_lasso_NOX,
    param_distributions=paramaters_l_NOX,
    scoring='neg_mean_squared_error',  # Specify your scoring metric
    n_iter=4,  # Number of parameter settings that are sampled
    random_state=42,  # Set a random seed for reproducibility
    n_jobs=-1  # Use all available CPU cores
)

pipe_search_l_NOX.fit(X_train,y_train['NOX'])

pipe_search_l_NOX.best_params_

y_pred_NOX_l=pipe_search_l.predict(X_test)

mae_gbr_NOX_l= mean_absolute_error(y_test['NOX'].values.astype(float), y_pred_NOX_l)

mape_gbr_NOX_l=mean_absolute_percentage_error(y_test['NOX'].values.astype(float), y_pred_NOX_l)*100

mse_gbr_NOX_l=mean_squared_error(y_test['NOX'].values.astype(float), y_pred_NOX_l)

#%%












#%%





