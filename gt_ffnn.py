# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 08:48:25 2023

@author: marco
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,median_absolute_error,r2_score
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
#%%
gt_files=['gt_2011.csv','gt_2012.csv','gt_2013.csv','gt_2014.csv','gt_2015.csv']

columns_extract=['CO','NOX']

gt_completo=[pd.read_csv(file,delimiter=';') for file in gt_files]

gt_completo=pd.concat(gt_completo, axis=0, ignore_index=True)

gt_completo['CDP']=gt_completo['CDP']*1000

Y_target = [pd.read_csv(file,delimiter=';',usecols=columns_extract) for file in gt_files]

Y_target=pd.concat(Y_target, axis=0, ignore_index=True)

columns_extract2=['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']



X=[pd.read_csv(file,delimiter=';',usecols=columns_extract2) for file in gt_files]

X=pd.concat(X, axis=0, ignore_index=True) #vettore delle X




X_train,X_test,y_train,y_test = train_test_split(X,Y_target,test_size=0.2,random_state=0)

#%%
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%


# Step 3: Build the model
model = models.Sequential()

# Add input layer
model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))  # Input shape based on the number of features

# Add hidden layers
model.add(layers.Dense(units=160, activation='relu'))
model.add(layers.Dense(units=80, activation='relu'))

# Add output layer for regression (linear activation)
model.add(layers.Dense(units=1, activation='linear'))

# Step 4: Compile the model
model.compile(optimizer='adam',  # You can choose other optimizers like 'sgd' or 'rmsprop'
              loss='mean_squared_error',  # Use mean squared error for regression
              metrics=['mae'])  # Mean Absolute Error can be a useful metric for regression

# Step 5: Print the model summary
model.summary()
# Step 6: Train the model
#model.fit(X_train, y_train['CO'], epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test['CO'])
print(f'Test Mean Absolute Error: {test_mae}')

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs2', histogram_freq=1)


model.fit(X_train, y_train['CO'], epochs=10, batch_size=64,callbacks=[tensorboard_callback], validation_data=(X_test, y_test))

# Step 8: Make predictions
y_pred = model.predict(X_test)
#%%
mae_CO=mean_absolute_error(y_test['CO'].values.astype(float), y_pred)
mape_CO=mean_absolute_percentage_error(y_test['CO'].values.astype(float), y_pred)*100
mse_CO=mean_squared_error(y_test['CO'].values.astype(float), y_pred)
made_CO=median_absolute_error(y_test['CO'].values.astype(float), y_pred)*100
r_2_CO=r2_score(y_test['CO'].values.astype(float), y_pred)
#model.save('regression_model.h5')
#%%



