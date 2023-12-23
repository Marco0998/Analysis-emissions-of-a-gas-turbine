# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:19:20 2023

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



gt_files=['gt_2011.csv','gt_2012.csv','gt_2013.csv','gt_2014.csv','gt_2015.csv']

columns_extract=['CO','NOX']

gt_completo=[pd.read_csv(file,delimiter=';') for file in gt_files]

gt_completo=pd.concat(gt_completo, axis=0, ignore_index=True)

gt_completo['CDP']=gt_completo['CDP']*1000


gt_completo['GTEP']=gt_completo['GTEP']*100


Y_target = [pd.read_csv(file,delimiter=';',usecols=columns_extract) for file in gt_files]

Y_target=pd.concat(Y_target, axis=0, ignore_index=True)

Y_target=Y_target.iloc[:7500,:2]

columns_extract2=['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']



X=[pd.read_csv(file,delimiter=';',usecols=columns_extract2) for file in gt_files]

X=pd.concat(X, axis=0, ignore_index=True) #vettore delle X

X=X.iloc[:7500,:9]


X_train,X_test,y_train,y_test = train_test_split(X,Y_target,test_size=0.2,random_state=0)


#%%


pca_svr_NOX=PCA(n_components=7)

import time

start_time = time.time()


pipe_svc_NOX=Pipeline([('robust_scaler_NOX',RobustScaler()),('pca_svr_NOX',pca_svr_NOX),('svr_NOX',svm.SVR(C=1e4,gamma=1e-2))])

# parameters_svr_NOX= {'svr_NOX__C':[1e4],'svr_NOX__gamma' : [1e-2],'pca_svr_NOX__n_components':[7]}


svr_search_NOX = pipe_svc_NOX
# RandomizedSearchCV(
#     estimator=pipe_svc_NOX,
#     param_distributions=parameters_svr_NOX,
#     scoring='max_error',  # Specify your scoring metric
#     n_iter=1,  # Number of parameter settings that are sampled
#     random_state=42,  # Set a random seed for reproducibility
#     n_jobs=-1  # Use all available CPU cores
# )

svr_search_NOX.fit(X_train,y_train['NOX'])


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

#%%
y_pred_NOX_svr= svr_search_NOX.predict(X_test)

mae_gbr_NOX_svr= mean_absolute_error(y_test['NOX'].values.astype(float), y_pred_NOX_svr)

mape_gbr_NOX_svr=mean_absolute_percentage_error(y_test['NOX'].values.astype(float), y_pred_NOX_svr)*100

mse_gbr_NOX_svr=mean_squared_error(y_test['NOX'].values.astype(float), y_pred_NOX_svr)

made_NOX=median_absolute_error(y_test['NOX'].values.astype(float), y_pred_NOX_svr)

r_2_NOX=r2_score( y_test['NOX'].values.astype(float), y_pred_NOX_svr)

#%%








#%%



pca_svr_CO=PCA(n_components=6)

start_time1 = time.time()

pipe_svr_CO=Pipeline([('pca_svr_CO',pca_svr_CO),('svr_CO',svm.SVR(C=1e4,gamma=1e-3))])

#parameters_svr_CO= {'svr_CO__C':[1e3,1e4,1e5],'svr_CO__gamma' : [1e-3,1e-4]}



# svr_search_CO = RandomizedSearchCV(
#     estimator=pipe_svr_CO,
#     param_distributions=parameters_svr_CO,
#     scoring='max_error',  # Specify your scoring metric
#     n_iter=2,  # Number of parameter settings that are sampled
#     random_state=42,  # Set a random seed for reproducibility
#     n_jobs=-1  # Use all available CPU cores
# )

svr_search_CO=pipe_svr_CO

svr_search_CO.fit(X_train,y_train['CO'])

end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(f"Elapsed time: {elapsed_time1} seconds")

#%%

y_pred_CO_svr= svr_search_CO.predict(X_test)

mae_gbr_CO_svr= mean_absolute_error(y_test['CO'].values.astype(float), y_pred_CO_svr)

mape_gbr_CO_svr=mean_absolute_percentage_error(y_test['CO'].values.astype(float), y_pred_CO_svr)*100

mse_gbr_CO_svr=mean_squared_error(y_test['CO'].values.astype(float), y_pred_CO_svr)

made_CO=median_absolute_error(y_test['CO'].values.astype(float), y_pred_CO_svr)*100

r_2_CO=r2_score(y_test['CO'].values.astype(float), y_pred_CO_svr)







#%%
