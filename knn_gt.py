# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:24:39 2023

@author: marco
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
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
import tensorflow.python as tf 

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




X_train,X_test,y_train,y_test = train_test_split(X,Y_target,test_size=0.2,random_state=0)


#%%

pca_knn_NOX=PCA()

pipe_knn_NOX=Pipeline([('robust_scaler_NOX',RobustScaler()),('pca_knn_NOX',pca_knn_NOX),('knn_NOX',KNeighborsRegressor())])

parameters_knn_NOX= {'knn_NOX__n_neighbors':[15,20,25],'pca_knn_NOX__n_components':[5,6,7,8]}


knn_search_NOX = RandomizedSearchCV(
    estimator=pipe_knn_NOX,
    param_distributions=parameters_knn_NOX,
    scoring='neg_mean_squared_error',  # Specify your scoring metric
    n_iter=4,  # Number of parameter settings that are sampled
    random_state=42,  # Set a random seed for reproducibility
    n_jobs=-1  # Use all available CPU cores
)

knn_search_NOX.fit(X_train,y_train['NOX'])

knn_search_NOX.best_params_
#%%
y_pred_NOX_knn= knn_search_NOX.predict(X_test)

mae_gbr_NOX_knn= mean_absolute_error(y_test['NOX'].values.astype(float),y_pred_NOX_knn)

mape_gbr_NOX_knn=mean_absolute_percentage_error(y_test['NOX'].values.astype(float), y_pred_NOX_knn)*100

mse_gbr_NOX_knn=mean_squared_error(y_test['NOX'].values.astype(float), y_pred_NOX_knn)

made_NOX=median_absolute_error(y_test['NOX'].values.astype(float), y_pred_NOX_knn)



r_2_NOX=r2_score(y_test['NOX'].values.astype(float), y_pred_NOX_knn)


#%%
import time

start_time = time.time()


pca_knn_CO=PCA(n_components=7)

pipe_knn_CO=Pipeline([('robust_scaler_NOX',RobustScaler()),('pca_knn_CO',pca_knn_NOX),('knn_CO',KNeighborsRegressor(n_neighbors=25,n_jobs=-1))])

# parameters_knn_CO= {'knn_CO__n_neighbors':[15,20,25],'pca_knn_CO__n_components':[5,6,7,8]}


# knn_search_CO = RandomizedSearchCV(
#     estimator=pipe_knn_CO,
#     param_distributions=parameters_knn_CO,
#     scoring='neg_mean_squared_error',  # Specify your scoring metric
#     n_iter=4,  # Number of parameter settings that are sampled
#     random_state=42,  # Set a random seed for reproducibility
#     n_jobs=-1  # Use all available CPU cores
# )

knn_search_CO=pipe_knn_CO

knn_search_CO.fit(X_train,y_train['CO'])



end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# knn_search_CO.best_params_
#%%


y_pred_CO_knn= knn_search_CO.predict(X_test)

mae_gbr_CO_knn= mean_absolute_error(y_test['CO'].values.astype(float),y_pred_CO_knn)

mape_gbr_CO_knn=mean_absolute_percentage_error(y_test['CO'].values.astype(float), y_pred_CO_knn)*100

mse_gbr_CO_knn=mean_squared_error(y_test['CO'].values.astype(float), y_pred_CO_knn)

made_CO=median_absolute_error(y_test['CO'].values.astype(float), y_pred_CO_knn)*100
r_2_CO=r2_score(y_test['CO'].values.astype(float), y_pred_CO_knn)

#%%










#%%