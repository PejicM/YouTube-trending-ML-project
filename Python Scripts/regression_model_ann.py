# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:12:23 2020

@author: Marko Pejić
"""


#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import scipy.stats as stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#%% Data Loading
videos_df = pd.read_pickle('US_trending.pkl')

#%% Missing data
# ovde mozda razmisliti o necemu boljem (srednja vrednost po kategoriji i slicno)
videos_df['positive_sentiment'] = videos_df['positive_sentiment'].fillna(videos_df['positive_sentiment'].mean())
videos_df['negative_sentiment'] = videos_df['negative_sentiment'].fillna(videos_df['negative_sentiment'].mean())
videos_df['neutral_sentiment'] = videos_df['neutral_sentiment'].fillna(videos_df['neutral_sentiment'].mean())
print(videos_df.isna().sum())

#%% Scale numerical features
numerical_features = ['view_count', 'dislikes', 'comment_count', 'positive_sentiment', 'negative_sentiment']
scaler = StandardScaler()
videos_df[numerical_features] = scaler.fit_transform(videos_df[numerical_features])

#%% Create X and y arrays
X = videos_df[['view_count', 'dislikes', 'comment_count', 'positive_sentiment', 'negative_sentiment', 'category_id']]
y = videos_df['likes'].values

#%% One hot encoding for category feature
onehotencoder = OneHotEncoder(categorical_features = [5]) 
X = onehotencoder.fit_transform(X).toarray() 

#%% Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

#%% Create ANN model
features = X.shape[1]

def create_regressor_ANN():
    model = Sequential()
    model.add(Dense(units=64, input_dim=features, activation='relu'))
    model.add(Dense(units=32))
    model.add(Dense(units=1))
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    return model

#%% Training of ANN model
model = create_regressor_ANN()
model.fit(X_train, y_train, batch_size=5, epochs=100, validation_split=0.1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

#%% Make predictions
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)

