# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:52:23 2023

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
#gt_2011= pd.read_csv('gt_2011.csv',delimiter=';')
#gt_2012= pd.read_csv('gt_2012.csv',delimiter=';')
#gt_2013= pd.read_csv('gt_2013.csv',delimiter=';')
#gt_2014= pd.read_csv('gt_2014.csv',delimiter=';')
#gt_2015= pd.read_csv('gt_2015.csv',delimiter=';')

#%% Primo metodo di estrazione dati
"Viene creata un unica matrice X ed Y combinando tutti i dati"

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





# cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

# def magnify():
#     return [dict(selector="th",
#                  props=[("font-size", "7pt")]),
#             dict(selector="td",
#                  props=[('padding', "0em 0em")]),
#             dict(selector="th:hover",
#                  props=[("font-size", "12pt")]),
#             dict(selector="tr:hover td:hover",
#                  props=[('max-width', '200px'),
#                         ('font-size', '12pt')])
# ]
# corr = gt_completo.corr()
# corr.style.background_gradient(cmap, axis=1)\
#     .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
#     .set_caption("Hover to magify")\
#     .set_table_styles(magnify())


#%%
pca_NOX=PCA(n_components=8)
import time

start_time = time.time()


pipe_gbr_NOX=Pipeline([('robust_scaler_NOX',RobustScaler()),('pca_NOX',pca_NOX),('gbr_NOX',GradientBoostingRegressor())])

pipe_gbr_NOX.fit(X_train,y_train['NOX'])


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


y_pred_NOX= pipe_gbr_NOX.predict(X_test)

mae_gbr_NOX= mean_absolute_error(y_test['NOX'].values.astype(float), y_pred_NOX)

mape_gbr_NOX=mean_absolute_percentage_error(y_test['NOX'].values.astype(float), y_pred_NOX)*100

mse_gbr_NOX=mean_squared_error(y_test['NOX'].values.astype(float), y_pred_NOX)



made_NOX=median_absolute_error(y_test['NOX'].values.astype(float), y_pred_NOX)*100

r_2_NOX=r2_score(y_test['NOX'].values.astype(float), y_pred_NOX)

#%%



pca_CO=PCA(n_components=7)

start_time = time.time()


pipe_gbr_CO=Pipeline([('robust_scaler_CO',RobustScaler()),('pca_CO',pca_CO),('gbr_CO',GradientBoostingRegressor())])



pipe_gbr_CO.fit(X_train,y_train['CO'])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


y_pred_CO= pipe_gbr_CO.predict(X_test)

mae_gbr_CO= mean_absolute_error(y_test['CO'].values.astype(float), y_pred_CO)

mape_gbr_CO=mean_absolute_percentage_error(y_test['CO'].values.astype(float), y_pred_CO)*100

mse_gbr_CO=mean_squared_error(y_test['CO'].values.astype(float), y_pred_CO)

made_CO=median_absolute_error(y_test['CO'].values.astype(float), y_pred_CO)*100

r_2_CO=r2_score(y_test['CO'].values.astype(float), y_pred_CO)

#%%
# pca.fit(X_train)

# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
# ax0.plot(
#     np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
# )
# ax0.set_ylabel("PCA explained variance ratio")

# ax0.axvline(
#     pca.n_components_,
#     linestyle=":",
#     label="n_components chosen",
# )
# ax0.legend(prop=dict(size=12))
# plt.show()

#%%
