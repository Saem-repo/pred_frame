#%%

# from cProfile import label
# from re import A
# from tabnanny import verbose
# from turtle import title
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
from PIL import Image
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


# 나중에 밑에 라이브러리 정리해야함!! 필요한 것들만 추려서,,

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
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
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

#%%

# 데이터를 받아서 하게 되면!!~~~ 기 수집했던 데이터는 따로 하지 않아도 됨!!
# 그래도 모르니깐,,, 데이터 올려놓고,, 데이터 받게 만들자!!
# 전역변수 및 시스템 환경 설정에 필요한 변수들 모아놓!!

global data
global fracs    

fracs = [0.2, 0.4, 0.6, 0.8, 1]
            

#%%

st.set_page_config(layout='wide',
                   page_icon='./smart_grid.png', 
                #    initial_sidebar_state='collapsed',
                   page_title='Ensembled Trasnfer Learning for Integrated Building Management System')

# streamlit UI design

pred_task = ['Building Energy', 'Individual Thermal Comfort', 'Natural Ventilation Rate']

#st.sidebar.image('smart_city.png', width=250)

st.sidebar.image('smart_city.png')

st.sidebar.header('For Prediction of Building Energy, Thermal Comfort, Natural Ventilation')
st.sidebar.markdown('Based on Knowledge Sharing AI')


menu = st.sidebar.radio(
    "",
    ("Introduction",
     "Exploratory Data Analysis (EDA)",
     "Development of Prediction Model"
     ),
)

# "Building Energy",
# "Individual Thermal Comfort",
# "Natural Ventilation"

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.markdown('''Hansaem Park | SSEL   
                    E-mail: saem@kaist.ac.kr  
                    Website: https://ssel.kaist.ac.kr/''')

if menu == 'Introduction':
    st.write('여기는 본 Framework의 목적 및 내 연구 결과(논문) 등을 정리해서 보여주는곳')
    
    st.write("""
             # 1st version of prediction framework
             """)

    st.markdown("""
                This app performs prediction tasks based on transfer learning with different ensembled strategies.
                * **Transfer learning :** only performs fine-tuned layers on small amount of datasets in target domain
                * **Ensembled transfer learning :** performs ensembled neural networks to develop the pre-trained model with datasets in source domains
                * **Hybrid ensembled transfer learning :** performs ensembeld neural networks to develop the pre-trained model and also apply ensembled strategies when model transfer
                """)

    img = Image.open('C:\\Users\\ssel\\Desktop\\Experiment\\Prediction Framework\\Research Outline.png')
    st.image(img)
    
    

elif menu == "Exploratory Data Analysis (EDA)":
    # st.write('여기는 학위논문에서 사용한 데이터 셋들에 대한 통계치 및 사용 변수들 정리해서 시각화 해주기')
    
    st.markdown(''' ### Three Types of Experimental Datasets  ''')
    
    eda_menu= st.radio(
        "",
        ("Building Energy", "Thermal Comfort","Natural Ventilation Rate"),
    )
    if eda_menu == "Building Energy":
        
        energy_sum = pd.read_csv('./#Data/building_energy/energy_summer.csv')
        energy_win = pd.read_csv('./#Data/building_energy/energy_winter.csv')
        
        energy_sum.loc[energy_sum['applied_engi']==0,'applied_engi'] = np.NaN

        energy_win.loc[energy_win['applied_engi']>=2000, 'applied_engi'] = np.NaN
        energy_win.loc[energy_win['applied_engi']==0,'applied_engi'] = np.NaN

        energy_sum.loc[energy_sum['mir_dorm']>=200, 'mir_dorm'] = np.NaN
        energy_sum.loc[energy_sum['mir_dorm']==0, 'mir_dorm'] = np.NaN

        energy_win.loc[energy_win['mir_dorm']>=400, 'mir_dorm'] = np.NaN
        energy_win.loc[energy_win['mir_dorm']==0, 'mir_dorm'] = np.NaN

        energy_sum = energy_sum.fillna(method='ffill')
        energy_win = energy_win.fillna(method='ffill')
        
        def data_extraction (df, bldg) :
            temp_df = df.iloc[:,:9]
            temp_df[bldg] = df.loc[:,bldg]
            
            return temp_df
        
        energy_sum_w1 = data_extraction(energy_sum, 'applied_engi')
        energy_sum_w6 = data_extraction(energy_sum, 'mir_dorm')
        
        energy_win_w1 = data_extraction(energy_win, 'applied_engi')
        energy_win_w6 = data_extraction(energy_win, 'mir_dorm')
        
        st.markdown(''' ### Source and Target Buildings ''')

        with st.expander("Statistics of source building"):
            st.write('Source buildings (in Summer)')
            st.table(energy_sum.describe())
        
        with st.expander("Statistics of target building"):
            st.write('Target building (W1 in Winter: Case 1)')
            st.table(energy_win_w1.describe())
            
            st.write('Target building (W6 in Summer: Case 2)')
            st.table(energy_sum_w6.describe())
            
            st.write('Target building (W6 in Winter: Case 3)')
            st.table(energy_win_w6.describe())
        
        def lineplot(df, df1, data_cols) :
            # plt.style.use('dark_background')
            filtered_df_sum = df.loc[:,data_cols]
            filtered_df_win = df1.loc[:,data_cols]


            # print(filtered_df.columns[0])
            fig = plt.figure(figsize=(10,7))
            plt.plot(filtered_df_sum, color='r', label='Summer')
            plt.plot(filtered_df_win, color='y', label='Winter')
            
            if data_cols == 'temp' :
                plt.ylabel('Outdoor Temperature $(\N{DEGREE SIGN}C)$')
            elif data_cols == 'humidity' :
                plt.ylabel('Outdoor Relative Humidity Difference (%)')
            else :
                plt.ylabel('Energy Consumption $(kWh)$')
            
            plt.legend(loc='best')
            
            return st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### Energy profile of W1 in different season')
            lineplot(energy_sum_w1, energy_win_w1,'applied_engi')
            
        with col2:
            st.markdown('### Energy profile of W6 in different season')
            lineplot(energy_sum_w6, energy_win_w6,'mir_dorm')

        col3, col4 = st.columns(2)            
        with col3:
            st.markdown('### Temperature')
            lineplot(energy_sum_w6, energy_win_w6,'temp')

        with col4:
            st.markdown('### Humidity')
            lineplot(energy_sum_w6, energy_win_w6,'humidity')

                
    elif eda_menu == "Thermal Comfort":
        
        
        st.write('각 사람들 클래스 레이블 및 실내 온습도와 각 생리학 정보 투영')
        
        phs = pd.read_csv('C:\\Users\\ssel\\Desktop\\Experiment\\Prediction Framework\\#Data\\ptc\\3scaled_PHS.csv')
        pdy = pd.read_csv('C:\\Users\\ssel\\Desktop\\Experiment\\Prediction Framework\\#Data\\ptc\\3scaled_PDY.csv')
        khm = pd.read_csv('C:\\Users\\ssel\\Desktop\\Experiment\\Prediction Framework\\#Data\\ptc\\3scaled_KHM.csv')
        
        
        phs_df_rev = phs
        pdy_df_rev = pdy
        khm_df_rev = khm

        phs_df_rev['Who'] = 'Occupant A'
        pdy_df_rev['Who'] = 'Occupant B'
        khm_df_rev['Who'] = 'Occupant C'

        total_df = pd.concat([phs_df_rev,pdy_df_rev,khm_df_rev], axis=0)
        
        def extract_value_3_scaled (df) :
            df_count = [
            df.loc[(df.TSV == -1), 'TSV'].count(),
            df.loc[(df.TSV == 0), 'TSV'].count(),
            df.loc[(df.TSV == 1), 'TSV'].count()
            ]

            df_rh = [
            round(df.loc[(df.TSV == -1), 'Rhin'].mean(), 2),
            round(df.loc[(df.TSV == 0), 'Rhin'].mean(), 2),
            round(df.loc[(df.TSV == 1), 'Rhin'].mean(), 2)
            ]
            
            df_t = [
            round(df.loc[(df.TSV == -1), 'Tin'].mean(), 2),
            round(df.loc[(df.TSV == 0), 'Tin'].mean(), 2),
            round(df.loc[(df.TSV == 1), 'Tin'].mean(), 2)
            ]
            df_label = ['Cold', 'Neutral', 'Hot']
            
            
            new_df = pd.DataFrame([df_count, df_t, df_rh], index=['count','T','RH'], columns=df_label)
            new_df = new_df.fillna(0)
            new_df = new_df.T
            
            return new_df
        
        def boxplot(df, col_name):
            fig = plt.figure(figsize=(10,7))
            sns.boxplot(x=df.Who, y=df[col_name], data=df, showmeans=True,
                        meanprops={"marker":"o", "markerfacecolor":"white", 
                                   "markeredgecolor":"black",
                                   "markersize":"10"})
            
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel("Occupants", fontsize=15)
            plt.ylabel(col_name, fontsize=15)
            
            return st.pyplot(fig)
        
        st.markdown(''' ### Source and Target Individuals ''')

        with st.expander("Statistics of source person"):
            st.write('Source person (Measured in Winter: PHS)')
            st.table(phs.describe())
        
        with st.expander("Statistics of target person"):
            st.write('Target person (Different thermal environment: Case 1)')
            st.table(pdy.describe())
            
            st.write('Target person (Similar thermal environment: Case 2)')
            st.table(khm.describe())
            
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('### BVP')
            boxplot(total_df, 'BVP')
            
        with col2:
            st.markdown('### EDA')
            boxplot(total_df, 'EDA')
            
        with col3:
            st.markdown('### ST')
            boxplot(total_df, 'TEMP')
            
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown('### HR')
            boxplot(total_df, 'HR')
            
        with col5:
            st.markdown('### Tin')
            boxplot(total_df, 'Tin')
            
        with col6:
            st.markdown('### RHin')
            boxplot(total_df, 'Rhin')
            
    elif eda_menu == "Natural Ventilation Rate":
        st.write('가을 및 여름 데이터 통계량과 수집일에 따른 자연환기량 ')
        
        nv_df_sum = pd.read_csv("C:\\Users\\ssel\\Desktop\\Experiment\\Prediction Framework\\#Data\\nvr\\20220328_sum_vars.csv")
        nv_df_fall = pd.read_csv("C:\\Users\\ssel\\Desktop\\Experiment\\Prediction Framework\\#Data\\nvr\\20220328_fall_vars.csv")

        nv_df_sum_time = pd.read_csv("C:\\Users\\ssel\\Desktop\\Experiment\\Prediction Framework\\#Data\\nvr\\20220328_sum_data.csv")
        nv_df_fall_time = pd.read_csv("C:\\Users\\ssel\\Desktop\\Experiment\\Prediction Framework\\#Data\\nvr\\20220328_fall_data.csv")
    
        nv_df_sum_time['Time'] = pd.to_datetime(nv_df_sum_time['Time'], format = '%Y-%m-%d %H:%M', errors= 'raise')
        nv_df_fall_time['time'] = pd.to_datetime(nv_df_fall_time['time'], format = '%H:%M:%S', errors= 'raise')
        # date = nv_df_sum_rev['Time'].dt.strftime("%m-%d")
        hour_sum = nv_df_sum_time['Time'].dt.hour
        hour_fall = nv_df_fall_time['time'].dt.hour
        nv_df_sum_time['hour'] = hour_sum
        nv_df_fall_time['hour'] = hour_fall
        
        nv_df_sum_time_rev = nv_df_sum_time.loc[(nv_df_sum_time['hour'] >= 10) & (nv_df_sum_time['hour'] < 19), :]
        nv_df_fall_time_rev = nv_df_fall_time.loc[(nv_df_fall_time['hour'] >= 10) & (nv_df_fall_time['hour'] < 19), :]
        
        
        def barplot(df) :
            # plt.style.use('dark_background')
            # filtered_df_sum = df.loc[:,data_cols]
            # filtered_df_win = df1.loc[:,data_cols]

            # print(filtered_df.columns[0])
            fig = plt.figure(figsize=(10,7))
            sns.barplot(x=df.hour, y=df.NVR, data=df, estimator=np.mean, ci=95, capsize=0.3)
            # plt.ylim(0.0, 14.0)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel("Hour", fontsize=20)
            plt.ylabel("Averaged Natural Ventilation Rate $(m^3/m)$", fontsize=20)
            plt.tight_layout()
            
            return st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### Averaged natural ventilation rate of source office')
            barplot(nv_df_fall_time_rev)
            
        with col2:
            st.markdown('### Averaged natural ventilation rate of target office')
            barplot(nv_df_sum_time_rev)
            
        st.markdown(''' ### Source and Target Buildings ''')

        with st.expander("Statistics of source building"):
            st.write('Source building (Munji Campus Room #106)')
            st.table(nv_df_fall.describe())
        
        with st.expander("Statistics of target building"):
            st.write('Target building (W16 Room #501)')
            st.table(nv_df_sum.describe())
            
            
elif menu == "Development of Prediction Model" :
        
    st.title(""" Enesembled Transferable Predicitve Models on Building Tasks """)
    
    # task_menu= st.radio(
    #     "",
    #     ("Building Energy", "Individual Thermal Comfort","Natural Ventilation Rate"),
    # )
    
    
    tasks = ["Building Energy", "Individual Thermal Comfort","Natural Ventilation Rate"]
    task_menu = st.selectbox("", tasks, index=0)


    #("Standard Transfer Learning", "Ensembled Transfer Learning","Hybrid Ensembled Transfer Learning"),

    if task_menu == "Building Energy":
        
        # st.write('건물 에너지 예측 모델 만드는 곳')
        
        st.markdown("""
                This app performs prediction tasks based on transfer learning with different ensembled strategies.
                * **Transfer learning :** only performs fine-tuned layers on small amount of datasets in target domain
                * **Ensembled transfer learning :** performs ensembled neural networks to develop the pre-trained model with datasets in source domains
                * **Hybrid ensembled transfer learning :** performs ensembeld neural networks to develop the pre-trained model and also apply ensembled strategies when model transfer
                """)
      
        model_type = st.radio(
            "",
            ("STL", "ETL","HETL"),
        )
        
        # st.write('Building energy에 대해 각 모델 전이 전략에 따른 예측 결과들 나타내는곳')
    
        if model_type == 'STL' :
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot
            from pred_method import errorplot
            
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            
            if st.button("Load Data & Generating Model"):
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                # print(cols)
                with st.spinner('Wating...'):
                    model, stl_pred, stl_act = cnn_reg(data, target, 2, 'STL_Energy')
                    
                    stl_mape = mean_absolute_percentage_error(stl_act, stl_pred)
                    stl_r2 = r2_score(stl_act, stl_pred)
                    stl_cv_rmse = cv_rmse(stl_act, stl_pred)
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[stl_r2, stl_cv_rmse, stl_mape]], columns=['R-squared', 'CV-RMSE', 'MAPE'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        # st.write(error_df)
                        errorplot(error_df)
                    with col2 :
                        regplot(stl_act, stl_pred)
                    
            #if st.button("Generating AI Model"):
            
                
        elif model_type == 'ETL' :
            
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot, errorplot
                    # Raw data 
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            if st.button("Load Data & Generating Model"):
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                # st.write(frac)
                
                with st.spinner('Wating...'):
                    model, stl_pred, stl_act = cnn_reg(data, target, 2, 'STL_Energy')
                    etl_model, etl_pred, etl_act = etl(model, data, target, 2, 'ETL_Energy', frac)
                    
                    etl_mape = mean_absolute_percentage_error(etl_act, etl_pred)
                    etl_r2 = r2_score(etl_act, etl_pred)
                    etl_cv_rmse = cv_rmse(etl_act, etl_pred)
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[etl_r2, etl_cv_rmse, etl_mape]], columns=['R-squared', 'CV-RMSE', 'MAPE'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        errorplot(error_df)
                        
                    with col2 :
                        regplot(etl_act, etl_pred)
                    
            #if st.button("Generating AI Model"):
            
        elif model_type == 'HETL' :
            
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot, stacked_model, fit_stacked_model
            from pred_method import errorplot
            
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            if st.button("Load Data & Generating Model"):
                
                    # Raw data 
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                    
                with st.spinner('Wating...'):
                    # model, stl_pred, stl_act = cnn_reg(data, target, 2, 'STL_Energy')
                    # etl_model, etl_pred, etl_act = etl(model, data, target, 2, 'ETL_Energy', frac)
                    # error = mean_absolute_percentage_error(stl_act, stl_pred)
                    
                    hetl_model, hetl_pred, hetl_act = stacked_model(data, target, 2, 'HETL_', 'reg')
                    
                    
                    hetl_mape = mean_absolute_percentage_error(hetl_act, hetl_pred)
                    hetl_r2 = r2_score(hetl_act, hetl_pred)
                    hetl_cv_rmse = cv_rmse(hetl_act, hetl_pred)
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[hetl_r2, hetl_cv_rmse, hetl_mape]], columns=['R-squared', 'CV-RMSE', 'MAPE'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        # st.write(error_df)
                        errorplot(error_df)
                    with col2 :
                        regplot(hetl_act, hetl_pred)
            
    #-----------------------------------------------------------------------------------------
    
    elif task_menu == "Individual Thermal Comfort":
        st.write('개별 열쾌적성 예측 모델 만드는 곳')
        st.markdown("""
                This app performs prediction tasks based on transfer learning with different ensembled strategies.
                * **Transfer learning :** only performs fine-tuned layers on small amount of datasets in target domain
                * **Ensembled transfer learning :** performs ensembled neural networks to develop the pre-trained model with datasets in source domains
                * **Hybrid ensembled transfer learning :** performs ensembeld neural networks to develop the pre-trained model and also apply ensembled strategies when model transfer
                """)
      
        model_type = st.radio(
            "",
            ("STL", "ETL","HETL"),
        )
        
        # st.write('Building energy에 대해 각 모델 전이 전략에 따른 예측 결과들 나타내는곳')
    
        if model_type == 'STL' :
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot, errorplot
            
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            if st.button("Load Data & Generating Model"):
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                # print(cols)
                with st.spinner('Wating...'):
                    model, stl_pred, stl_act = cnn_clf(data, target, 2, 'STL_ITC')
                    
                    stl_acc = accuracy_score(stl_act, stl_pred)
                    stl_f1 = f1_score(stl_act, stl_pred, average = 'weighted')
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[stl_acc, stl_f1]], columns=['Accuracy', 'F1-score'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        # st.write(error_df)
                        errorplot(error_df)
                    with col2 :
                        st.write(confusion_matrix(stl_act, stl_pred))
                    
            #if st.button("Generating AI Model"):
            
                
        elif model_type == 'ETL' :
            
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot, errorplot
                    # Raw data 
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            if st.button("Load Data & Generating Model"):
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                # st.write(frac)
                
                with st.spinner('Wating...'):
                    model, stl_pred, stl_act = cnn_clf(data, target, 2, 'STL_ITC')
                    etl_model, etl_pred, etl_act = etl(model, data, target, 2, 'ETL_ITC', frac)
                    
                    etl_acc = accuracy_score(etl_act, etl_pred)
                    etl_f1 = f1_score(etl_act, etl_pred)
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[etl_acc, etl_f1]], columns=['Accuracy', 'F1-score'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        errorplot(error_df)
                        
                    with col2 :
                        regplot(etl_act, etl_pred)
                    
            #if st.button("Generating AI Model"):
            
        elif model_type == 'HETL' :
            
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot, stacked_model, fit_stacked_model
            from pred_method import errorplot
            
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            if st.button("Load Data & Generating Model"):
                
                    # Raw data 
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                    
                with st.spinner('Wating...'):
                    # model, stl_pred, stl_act = cnn_reg(data, target, 2, 'STL_Energy')
                    # etl_model, etl_pred, etl_act = etl(model, data, target, 2, 'ETL_Energy', frac)
                    # error = mean_absolute_percentage_error(stl_act, stl_pred)
                    
                    hetl_model, hetl_pred, hetl_act = stacked_model(data, target, 2, 'HETL_', 'clf')
                    
                    
                    hetl_acc = accuracy_score(hetl_act, hetl_pred)
                    hetl_f1 = f1_score(hetl_act, hetl_pred)
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[hetl_acc, hetl_f1]], columns=['Accuracy', 'F1-score'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        # st.write(error_df)
                        errorplot(error_df)
                    with col2 :
                        regplot(hetl_act, hetl_pred)


    #-----------------------------------------------------------------------------------------
    
    elif task_menu == "Natural Ventilation Rate":
        # st.write('자연환기 예측 모델 만드는 곳')
        st.markdown("""
                This app performs prediction tasks based on transfer learning with different ensembled strategies.
                * **Transfer learning :** only performs fine-tuned layers on small amount of datasets in target domain
                * **Ensembled transfer learning :** performs ensembled neural networks to develop the pre-trained model with datasets in source domains
                * **Hybrid ensembled transfer learning :** performs ensembeld neural networks to develop the pre-trained model and also apply ensembled strategies when model transfer
                """)
      
        model_type = st.radio(
            "",
            ("STL", "ETL","HETL"),
        )
        
        # st.write('Building energy에 대해 각 모델 전이 전략에 따른 예측 결과들 나타내는곳')
    
        if model_type == 'STL' :
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot, errorplot
            
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            if st.button("Load Data & Generating Model"):
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                # print(cols)
                with st.spinner('Wating...'):
                    model, stl_pred, stl_act = cnn_reg(data, target, 2, 'STL_Energy')
                    
                    stl_mape = mean_absolute_percentage_error(stl_act, stl_pred)
                    stl_r2 = r2_score(stl_act, stl_pred)
                    stl_cv_rmse = cv_rmse(stl_act, stl_pred)
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[stl_r2, stl_cv_rmse, stl_mape]], columns=['R-squared', 'CV-RMSE', 'MAPE'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        # st.write(error_df)
                        errorplot(error_df)
                    with col2 :
                        regplot(stl_act, stl_pred)
                    
            #if st.button("Generating AI Model"):
            
                
        elif model_type == 'ETL' :
            
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot, errorplot
                    # Raw data 
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            if st.button("Load Data & Generating Model"):
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                # st.write(frac)
                
                with st.spinner('Wating...'):
                    model, stl_pred, stl_act = cnn_reg(data, target, 2, 'STL_Energy')
                    etl_model, etl_pred, etl_act = etl(model, data, target, 2, 'ETL_Energy', frac)
                    
                    etl_mape = mean_absolute_percentage_error(etl_act, etl_pred)
                    etl_r2 = r2_score(etl_act, etl_pred)
                    etl_cv_rmse = cv_rmse(etl_act, etl_pred)
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[etl_r2, etl_cv_rmse, etl_mape]], columns=['R-squared', 'CV-RMSE', 'MAPE'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        errorplot(error_df)
                        
                    with col2 :
                        regplot(etl_act, etl_pred)
                    
            #if st.button("Generating AI Model"):
            
        elif model_type == 'HETL' :
            
            from pred_method import cnn_reg, cnn_clf, etl, cv_rmse, regplot, stacked_model, fit_stacked_model
            from pred_method import errorplot
            
            uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception as e:
                    print(e)
                    data = pd.read_excel(uploaded_file)
            
            frac_col, ensem_num_col = st.columns([2, 2])
                    
            with frac_col :
                st.subheader('Choose the ratio of fine-tuned data')
                frac = st.slider("", 
                            0.0, 1.0,
                            0.2)
            with ensem_num_col :
                st.subheader('Choose the number of ensembled networks')
                ensem_num = st.slider("", 
                            0, 10,
                            2)
            
            if st.button("Load Data & Generating Model"):
                
                    # Raw data 
                st.dataframe(data)
                cols = data.columns
                target = cols[-1]
                    
                with st.spinner('Wating...'):
                    # model, stl_pred, stl_act = cnn_reg(data, target, 2, 'STL_Energy')
                    # etl_model, etl_pred, etl_act = etl(model, data, target, 2, 'ETL_Energy', frac)
                    # error = mean_absolute_percentage_error(stl_act, stl_pred)
                    
                    hetl_model, hetl_pred, hetl_act = stacked_model(data, target, 2, 'HETL_', 'reg')
                    
                    
                    hetl_mape = mean_absolute_percentage_error(hetl_act, hetl_pred)
                    hetl_r2 = r2_score(hetl_act, hetl_pred)
                    hetl_cv_rmse = cv_rmse(hetl_act, hetl_pred)
                    #st.write(error)
                    
                    error_df = pd.DataFrame([[hetl_r2, hetl_cv_rmse]], columns=['R-squared', 'CV-RMSE'])
                    
                    col1, col2 = st.columns([2, 2])
                    
                    with col1 :
                        # st.write(error_df)
                        errorplot(error_df)
                    with col2 :
                        regplot(hetl_act, hetl_pred)

        
        
        
        
        
        



