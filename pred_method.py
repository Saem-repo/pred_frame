import tensorflow as tf
import numpy as np
import pandas as pd


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, LSTM

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import probplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.svm import SVC
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
from sklearn.linear_model import LinearRegression
import pickle
from tensorflow.keras.models import load_model
import os
import sys
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import joblib
import pickle
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Embedding, Flatten
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import random
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras 
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.utils import plot_model
# from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score
import streamlit as st

#from mlxtend.regressor import StackingCVRegressor


def errorplot(df) :
    fig = plt.figure(figsize=(10,7))
    sns.barplot(data=df, estimator=np.mean, ci=95, capsize=0.3)
    # plt.ylim(0.0, 14.0)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Evaluation Metric", fontsize=20)
    plt.ylabel("Error rate", fontsize=20)
    plt.tight_layout()
    
    return st.pyplot(fig)


def regplot(act, preds) :
    # print(filtered_df.columns[0])
    fig = plt.figure(figsize=(10,7))
    sns.regplot(act, preds, fit_reg=True, line_kws={'color': 'red'})
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    #plt.legend(loc='best')
    
    return st.pyplot(fig)

def cv_rmse(y_test, y_pred) :
    return round(np.sqrt(mean_squared_error(y_test, y_pred))/np.mean(y_test), 2)

def mape (act, pred) :
    return round(np.mean(np.abs((act-pred)/act)), 2)

def RF (X_train, X_test, Y_train, Y_test, name) :
    params = {'n_estimators' : [100, 500, 1000, 2000, 3000],
          'max_depth' : [10, 15, 20, 25, 30],
          'min_samples_leaf' : [10, 15, 20, 25],
          'min_samples_split' : [10, 15, 20, 25]
         }


    rf = RandomForestRegressor(random_state=0)
    rf_grid_cv = GridSearchCV(rf, param_grid=params, cv = 10, n_jobs= -1, verbose=2)
    stime = time.time()
    rf_grid_cv.fit(X_train, Y_train)
    print("Time for RF fitting: %.3f" % (time.time() - stime))
    rf_predict = rf_grid_cv.predict(X_test)

#     print('--------------------------------')
#     print(name+'_최적 하이퍼 파라미터 : ',grid_cv.best_params_)
#     print(name+'_예측 정확도 : {:.2f}'.format(grid_cv.best_score_))
#     print('--------------------------------')

    print(name+'_RF RMSE : ', round(np.sqrt(mean_squared_error(rf_predict, Y_test)), 2))
    print(name+'_RF MAE : ', round(mean_absolute_error(rf_predict, Y_test), 2))
    print(name+'_RF CV-RMSE : ', round(cv_rmse(rf_predict, Y_test), 2))
    print(name+'_RF MAPE : ', round(mape(rf_predict, Y_test), 2))
    print(name+'_RF R2 : ', round(r2_score(rf_predict, Y_test), 2))
    print('Finished')
    print('--------------------------------')
    
    return rf_grid_cv, rf_predict


# 일반 CNN, DNN, LSTM 모델 도출



# 그리고 

# 전통 기계학습 (SVR, RF, MLR 등) 모델 학습 함수

def data_processing (df, col, mode) :
    #  col 변수는 건물의 이름 !!
#  mode 값에 따라서 DNN으로 만들지 CNN으로 만들지 아님 LSTM 만들지 결정
#  mode == 1 : DNN, mode == 2 : CNN, mode == 3 : LSTM
    if mode == 1 :
        X, Y = df.iloc[:,:df.shape[1]-1], df.loc[:,col]
        
        # Split the data into train and test data:

        # 80% 훈련 데이터 분배 
#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1234)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

        
#         X_train = X[0:int(X.shape[0]*0.8)]
#         Y_train = Y[0:int(Y.shape[0]*0.8)]
#         X_test = X[int(X.shape[0]*0.8):]
#         Y_test = Y[int(Y.shape[0]*0.8):]
        
        # X_train, X_test = MinMaxScaler().fit_transform(X_train), MinMaxScaler().fit_transform(X_test)
        X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)
            
    else :
        X, Y = df.iloc[:,:df.shape[1]-1], df.loc[:,col]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#         X_train = X[0:int(X.shape[0]*0.8)]
#         Y_train = Y[0:int(Y.shape[0]*0.8)]
#         X_test = X[int(X.shape[0]*0.8):]
#         Y_test = Y[int(Y.shape[0]*0.8):]
        
        X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)
        
        # X_train, X_test = MinMaxScaler().fit_transform(X_train), MinMaxScaler().fit_transform(X_test)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, Y_train, X_test, Y_test 




def cnn_reg (df, target, mode, name, frac=None) :
    
    x_train, y_train, x_test, y_test = data_processing(df, target, mode)
    
    drop_out = 0.2
    epochs = 50
    batch = 16
    activation = 'relu'
    pool_size=1
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    print(x_train.shape, x_test.shape)
    
    model = Sequential()
    model.add(Conv1D(filters=200, kernel_size=10, activation=activation, input_shape=(x_train.shape[1], x_train.shape[2]))),
    model.add(MaxPooling1D(pool_size=pool_size)),
    model.add(Conv1D(filters=100, kernel_size=5, activation=activation, padding='same')),
    model.add(MaxPooling1D(pool_size=pool_size)),
    model.add(Conv1D(filters=50, kernel_size=5, activation=activation, padding='same')),
    model.add(MaxPooling1D(pool_size=pool_size)),
    model.add(Flatten()),
#     model.add(GlobalAveragePooling1D()),
#     model.add(GlobalMaxPooling1D()),
    model.add(Dense(50, activation=activation)),
    model.add(Dense(30, activation=activation)),
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mape'])
    
    
    filename = str(name)+'_cnn_model.h5'
    
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    
    stime = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch,
                                     validation_data=(x_test, y_test),
                                    #  callbacks=[early_stop],
                                     callbacks=[early_stop, checkpoint],
                                     verbose=0)
    
    com = time.time() - stime
    
    model_preds = model.predict(x_test)
    
    rmse_test_error = round(np.sqrt(mean_squared_error(y_test, model_preds.ravel())), 2)
    mae_test_error = round(mean_absolute_error(y_test, model_preds.ravel()), 2)
    mape_test_error = round(mape(np.array(y_test), model_preds.ravel()), 2)
    cv_rmse_test_error = round(cv_rmse(y_test, model_preds.ravel()), 2)
    r2_test_error = round(r2_score(y_test, model_preds.ravel()), 2)
    
#     print('Std Scaled dataset RMSE in test : ', rmse_test_error)
#     print('Std Scaled dataset MAE in test : ', mae_test_error)
#     print('Std Scaled dataset MAPE in test : ', mape_test_error)
#     print('Std Scaled dataset CV-RMSE in test ', cv_rmse_test_error)
#     print('Std Scaled dataset R2 in test ', r2_test_error)
#     print('-----------------------------------------------------------')
    
    print("Finised")
    
    return model, model_preds.ravel(), y_test


def encoding (Ytrain, Ytest) : # 신경망 사용할라면 라벨 one-hot encoding
    new_Y_train = to_categorical(Ytrain, num_classes=3)
    new_Y_test = to_categorical(Ytest, num_classes=3)
    
    return new_Y_train, new_Y_test


def cnn_clf (df, target, mode, name) :
    
    x_train, y_train, x_test, y_test = data_processing(df, target, mode)
    
    y_train_rev, y_test_rev = encoding(y_train, y_test)
    
    drop_out = 0.2
    epochs = 100
    batch = 32
    activation = 'relu'
    pool_size=1
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    print(x_train.shape, x_test.shape)
    print(y_train_rev.shape, y_test_rev.shape)
    
    model = Sequential()
    model.add(Conv1D(filters=200, kernel_size=3, activation=activation, input_shape=(x_train.shape[1], x_train.shape[2]))),
    model.add(MaxPooling1D(pool_size=pool_size)),
    model.add(Conv1D(filters=100, kernel_size=3, activation=activation, padding='same')),
    model.add(MaxPooling1D(pool_size=pool_size)),
    model.add(Conv1D(filters=50, kernel_size=3, activation=activation, padding='same')),
    model.add(MaxPooling1D(pool_size=pool_size)),
    model.add(Flatten()),
#     model.add(GlobalAveragePooling1D()),
#     model.add(GlobalMaxPooling1D()),
    model.add(Dense(50, activation=activation)),
    model.add(Dense(30, activation=activation)),
    model.add(Dense(y_train_rev.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    
    filename = str(name)+'_cnn_model.h5'
    
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', patience=30)
    
    stime = time.time()
    model.fit(x_train, y_train_rev, epochs=epochs, batch_size=batch,
                                     validation_data=(x_test, y_test_rev),
                                    #  callbacks=[early_stop],
                                     callbacks=[early_stop, checkpoint],
                                     verbose=0)
    
    com = time.time() - stime
    
    model_pred = model.predict(x_test)
    model_pred = np.argmax(model_pred, axis=1)
    
    
    print(model_pred.shape)
    print(model_pred[:10])
    
    # cm = confusion_matrix(y_test, model_pred)
    # report = classification_report(y_test, model_pred)
    
    print("Finised")
    
    return model, model_pred, y_test


# Stacked neural networks model!!

# Ensemble network to extract the features with random sampled datasets
def stacked_model(df, target, mode, name, type, idx=2) :
#     CNN을 가지고 sub-models들을 만드는 작업
    
    members = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    if type == 'reg' :
        
        for i in range(idx):
            cnn_model, pred, act = cnn_reg(df, target, mode, name)    
            members.append(cnn_model)        
            
        for i in range(len(members)):
            model = members[i]
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unuque layer name' issue
    #             layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
                
    #     ensemble multi-headed input
        ensemble_visible = [model.input for model in members]
    #     concat
        ensemble_outputs = [model.output for model in members]
        
        print(ensemble_outputs)
        
        model_concat = concatenate(ensemble_outputs)
        
        add_l = Dense(300, activation='relu')(model_concat)
        add_l = Dense(150, activation='relu')(add_l)
        add_l = BatchNormalization()(add_l)
        add_l = Dropout(0.2)(add_l)
        add_l = Dense(50, activation='relu')(add_l)
        add_l = BatchNormalization()(add_l)
        add_l = Dropout(0.2)(add_l)
        final_out = Dense(1, activation='linear')(add_l)
            
        final_model = Model(inputs=ensemble_visible, outputs=final_out)
        final_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        model, result, y_test = fit_stacked_model(final_model, df, target, mode, name)
        r2_test = round(r2_score(y_test, result.ravel()), 2)
        print('모델의 예측 정확도 성능 : {} %'.format(r2_test))
        
    elif type == 'clf' :
            
        for i in range(idx):
            cnn_model = cnn_clf(df, target, mode, name)    
            members.append(cnn_model)        
            
        for i in range(len(members)):
            model = members[i]
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unuque layer name' issue
    #             layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
                
    #     ensemble multi-headed input
        ensemble_visible = [model.input for model in members]
    #     concat
        ensemble_outputs = [model.output for model in members]
        
        print(ensemble_outputs)
        
        model_concat = concatenate(ensemble_outputs)
        
        add_l = Dense(300, activation='relu')(model_concat)
        add_l = BatchNormalization()(add_l)
        add_l = Dense(150, activation='relu')(add_l)
        add_l = BatchNormalization()(add_l)
        add_l = Dense(50, activation='relu')(add_l)
        final_out = Dense(1, activation='linear')(add_l)
        
        final_model = Model(inputs=ensemble_visible, outputs=final_out)
        final_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        
        model, result, y_test = fit_stacked_model(final_model, df, target, mode, name)
        acc = round(accuracy_score(y_test, result.ravel(), normalize=False), 2)
        print('모델의 예측 정확도 성능 : {} %'.format(acc))
    
    
    else : 
        print('예측 유형이 잘 못 돼었습니다. 유형 1: 회귀, 유형 2: 분류')
        
#     plot model
    # tf.keras.utils.plot_model(final_model, show_shapes=True, show_layer_names=True, dpi=600, to_file=str(idx)+'_final_model_cnn.png')
    
#     compile
    
    return model, result, y_test
    
        
def fit_stacked_model(model, df, target, mode, name, idx=None) :
    
    x_train, y_train, x_test, y_test = data_processing(df, target, mode)
    
    X_train = [x_train for _ in range(len(model.input))]
    X_test = [x_test for _ in range(len(model.input))]
    
    print(np.array(X_train).shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    
    model.fit(X_train, y_train, 
              epochs=300,
              batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[early_stop],
              verbose=0)
    
    pred = model.predict(X_test, verbose=0)
    
    model.save(name+'_ensembled_model.h5')    
    
    return model, pred, y_test


def etl (model, df, target, mode, name, frac=None) :
    
    #print(str(frac)+"_experimental trials")
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    filename = str(name)+'_etl_model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
            
    epochs = 100
    batch = 32

    if frac == 0.0 :
        x_train, y_train, x_test, y_test = data_processing(df, target, 2)
        
    else :
        n_samples = df.shape[0]*frac
        sample_df = df.iloc[:int(n_samples),:]
        x_train, y_train, x_test, y_test = data_processing(sample_df, target, 2)
    
    print('ETL model_'+ 'NVR' + '_data_shape :')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    # x_train = [x_train for _ in range(len(tl_ensembled_model.input))]
    # x_test = [x_test for _ in range(len(tl_ensembled_model.input))]
    
    early_stop = EarlyStopping(monitor='val_loss', patience=30)
    model.fit(x_train, y_train, epochs=epochs,
                    batch_size=batch, validation_data=(x_test, y_test),
                    callbacks=[early_stop, checkpoint], verbose=0)
    
    pred = model.predict(x_test)
    
    return model, pred, y_test
