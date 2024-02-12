#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
#################################
import matplotlib.pyplot as plt
import seaborn as sns
##############################################
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
#################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Conv1D, Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import warnings
warnings.filterwarnings('ignore')
sns.set_context('talk')
sns.set_style('darkgrid',{'axes.facecolor':'0.9'})
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sagu_gray = pd.read_csv('Bukit_Sagu_Gray.csv')
ubi_gray = pd.read_csv('Bukit_Ubi_Gray.csv')
panching_gray = pd.read_csv('Panching_Gray.csv')
semambu_gray = pd.read_csv('Semambu_Gray.csv')
lembing_gray = pd.read_csv('Sg_lembing_Gray.csv')


# In[3]:


sagu_gray['DATE'] = pd.to_datetime(sagu_gray['DATE'])
ubi_gray['DATE'] = pd.to_datetime(ubi_gray['DATE'])
panching_gray['DATE'] = pd.to_datetime(panching_gray['DATE'])
lembing_gray['DATE'] = pd.to_datetime(lembing_gray['DATE'])


# In[4]:


sagu_blue = pd.read_csv('Bukit_Sagu_Blue.csv')
ubi_blue = pd.read_csv('Bukit_Ubi_Blue.csv')
panching_blue = pd.read_csv('Panching_Blue.csv')
semambu_blue = pd.read_csv('Semambu_Blue.csv')
lembing_blue = pd.read_csv('Sg_Lembing_Blue.csv')


# In[5]:


sagu_blue['DATE'] = pd.to_datetime(sagu_blue['DATE'])
ubi_blue['DATE'] = pd.to_datetime(ubi_blue['DATE'])
panching_blue['DATE'] = pd.to_datetime(panching_blue['DATE'])
lembing_blue['DATE'] = pd.to_datetime(lembing_blue['DATE'])


# In[6]:


sagu_gray = sagu_gray.set_index('DATE')
ubi_gray = ubi_gray.set_index('DATE')
panching_gray = panching_gray.set_index('DATE')
lembing_gray = lembing_gray.set_index('DATE')
###############################################
sagu_blue = sagu_blue.set_index('DATE')
ubi_blue = ubi_blue.set_index('DATE')
panching_blue = panching_blue.set_index('DATE')
lembing_blue = lembing_blue.set_index('DATE')


# # Sampling and Training

# In[7]:


class Sampling_Training:
    def __init__(self,n_steps=7,train_size=0.8):
        self.n_steps = n_steps
        self.train_size = train_size
        self.scaler = MinMaxScaler()
    def processing(self,series):
        scaler = MinMaxScaler()
        log_series = self.log_function(series)
        train , test = self.Train_test_split(log_series,self.train_size)
        Train = self.scaler.fit_transform(np.array(train).reshape(-1,1))
        Test = self.scaler.fit_transform(np.array(test).reshape(-1,1))
        x_train , y_train = self.sampling(Train, self.n_steps)
        x_test , y_test = self.sampling(Test, self.n_steps)
        X_train = x_train.reshape((x_train.shape[0],x_train.shape[1], 1))
        X_test = x_test.reshape((x_test.shape[0],x_test.shape[1], 1))
        return X_train , y_train , X_test , y_test 
        
    def sampling(self,sequence, n_steps):
        X, Y = list(), list()
        for i in range(len(sequence)):
            sam = i + n_steps
            if sam > len(sequence)-1:
                break
            x, y = sequence[i:sam], sequence[sam]
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)
    def log_function(self,series):
        log_s = np.array([np.log1p(x) for x in series])
        return log_s
    
    def Train_test_split(self,series,train_size):
        n_train = int(series.shape[0]*self.train_size)
        train , test = series[:n_train] , series[n_train:]
        return train , test


# In[8]:


st = Sampling_Training()
X_train , y_train , X_test , y_test  = st.processing(sagu_gray['Total Grey'])


# In[9]:


print('The Shape of x train:',X_train.shape)
print('The Shape of x test:',X_test.shape)


# In[11]:


Y_test =  st.scaler.inverse_transform(np.array(y_test).reshape(-1,1))


# # Model Performance Functions

# In[12]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse
def calculate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    error =  {'mean_squared_error':round(mse, 3), ' mean_absolute_error':round(mae, 3), 
              'mean_absolute_percentage_error':round(mape, 3), 'root_mean_squared_error':round(rmse, 3)}
    return error


# # LSTM model

# In[14]:


n_steps = 7
def lstm(x_train,y_train,n_steps=n_steps,epochs=40,batch_size=4,verbose=0):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,verbose=verbose)
    return model


# In[15]:


model_lstm = lstm(X_train,y_train,n_steps=n_steps,epochs=40,batch_size=4,verbose=0)


# In[17]:


lstm_ypred = model_lstm.predict(X_test)
lstm_pred =  st.scaler.inverse_transform(lstm_ypred)


# In[18]:


print('The Performance of LSTM Model:')
lstm_p = calculate_performance(Y_test,lstm_pred)
print(lstm_p)


# In[19]:


print('The Accuracy of LSTM Model :')
lstm_acc = r2_score(Y_test,lstm_pred)
print(lstm_acc)


# In[24]:


plt.figure(figsize=(16,4))
plt.title('Actual vs Predicted values of LSTM')
plt.plot(Y_test,label='Actual')
plt.plot(lstm_pred,label='predicted')
plt.legend()
plt.show()


# # ANN Model

# In[21]:


def ANN(x_train,y_train, input_dim =n_steps, epochs=60,batch_size=5,verbose=0):
    x_train = x_train.reshape(X_train.shape[0],X_train.shape[1])
    model = Sequential()
    model.add(Dense(64, activation = 'relu', input_dim = n_steps))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,verbose=verbose)
    return model


# In[22]:


model_ann = ANN(X_train,y_train, input_dim =n_steps, epochs=60,batch_size=4,verbose=0)


# In[23]:


x_test = X_test.reshape(X_test.shape[0],X_test.shape[1])


# In[26]:



ann_ypred = model_ann.predict(x_test)
ann_pred =  st.scaler.inverse_transform(ann_ypred)


# In[27]:


print('The Performance of ANN Model:')
ann_p = calculate_performance(Y_test,ann_pred)
print(ann_p)


# In[28]:


print('The Accuracy of ANN Model :')
ann_acc = r2_score(Y_test,ann_pred)
print(ann_acc)


# In[29]:


plt.figure(figsize=(16,4))
plt.title('Actual vs Predicted values of ANN')
plt.plot(Y_test,label='Actual')
plt.plot(ann_pred,label='predicted')
plt.legend()
plt.show()


# # CNN Model

# In[30]:


def CNN(x_train,y_train, input_shape =n_steps, epochs=60,batch_size=4,verbose=0):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation="relu", input_shape=(n_steps,1) ))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,verbose=verbose) 
    return model


# In[31]:


model_cnn = CNN(X_train,y_train, input_shape =n_steps, epochs=60,batch_size=4,verbose=0)


# In[33]:


cnn_ypred = model_cnn.predict(X_test)
cnn_pred =  st.scaler.inverse_transform(cnn_ypred)


# In[34]:


print('The Performance of CNN Model:')
cnn_p = calculate_performance(Y_test,cnn_pred)
print(cnn_p)


# In[35]:


print('The Accuracy of CNN Model :')
cnn_acc = r2_score(Y_test,cnn_pred)
print(cnn_acc)


# In[36]:


plt.figure(figsize=(16,4))
plt.title('Actual vs Predicted values of CNN')
plt.plot(Y_test,label='Actual')
plt.plot(cnn_pred,label='predicted')
plt.legend()
plt.show()


# # Future Days comparision

# In[37]:



def forecast_lstm_cnn( data , model, num_future_steps=30 , look_back=7):
    df = data.reshape(data.shape[0]*data.shape[1])
    prediction = df[-look_back:]
    for _ in range(num_future_steps):
        x = prediction[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction = np.append(prediction, out)
    pred =  st.scaler.inverse_transform(prediction.reshape(prediction.shape[0],1))
    Y_pred = np.array([np.exp(x) for x in pred])
    return Y_pred

def forecast_ann(data , model, num_future_steps=30 , look_back=7):
    df = data.reshape(data.shape[0]*data.shape[1])
    prediction = df[-look_back:]
    for _ in range(num_future_steps):
        x = prediction[-look_back:]
        x = x.reshape((1, look_back))
        out = model.predict(x)[0][0]
        prediction = np.append(prediction, out)
    pred =  st.scaler.inverse_transform(prediction.reshape(prediction.shape[0],1))
    Y_pred = np.array([np.exp(x) for x in pred])
    return Y_pred


# In[46]:


cnn_fc= forecast_lstm_cnn(X_test,model=model_cnn)
lstm_fc = forecast_lstm_cnn(X_test,model=model_lstm)
#####################################################
ann_fc = forecast_ann(X_test,model=model_ann)


# In[47]:


days = np.arange(1,30+1)


# In[49]:


plt.figure(figsize=(16,4))
plt.title('The Comparsion of CNN, LSTM and ANN Predictions')
plt.plot(days,cnn_fc[7:].flatten(),label='CNN')
plt.plot(days,lstm_fc[7:].flatten(),label='LSTM')
plt.plot(days,ann_fc[7:],label='ANN')
plt.legend()
plt.show()


# # Forecasting Steps

# <h3>Step 1:</h3>
# <p>Use Sampling_Training function To create a x_train , y_train , x_test , y_test data. Like below i did this. you just have to fit series in Sampling_Training.processing(series)</p>

# In[ ]:


X_train , y_train , X_test , y_test  = Sampling_Training.processing(sagu_gray['Total Grey'])


# <h3>Step 2:</h3>
# <p>In second step Use one of Three Models Like LST, CNN, and ANN and fit the x_train and y_train datasets.</p>

# In[ ]:


model_lstm_sagu_gray = lstm(X_train,y_train)


# <h3>Step 3:</h3>
# <p>In this step you have to forecast the future values. I have created two functions first if you want to forecast in LSTM or  CNN models you should use forecast_lstm_cnn and if you want to use ANN model to Forecast use only forecast_ann function. In both function you just have to put two parameters x_test and model in which you want forecasting</p>

# In[ ]:


# for LSTM AND CNN
stm_fc = forecast_lstm_cnn(X_test,model=model_lstm)

# FOR ANN
ann_fc = forecast_ann(X_test,model=model_ann)


# # LSTM forecasting of 3 Months

# In[50]:


X_train , y_train , X_test , y_test  = st.processing(sagu_gray['Total Grey'])
####################################
model_lstm_sagu_gray = lstm(X_train,y_train)
lstm_fc_sagu_gray = forecast_lstm_cnn(X_test,model=model_lstm_sagu_gray)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of Bukit_Sagu_Gray')
plt.plot(days,lstm_fc_sagu_gray[7:].flatten(),label='Bukit_Sagu_Gray',color='red')
plt.ylabel('Total Grey')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[51]:


X_train , y_train , X_test , y_test  = st.processing(ubi_gray['Total Grey'])
####################################
model_lstm_ubi_gray = lstm(X_train,y_train)
lstm_fc_ubi_gray = forecast_lstm_cnn(X_test,model=model_lstm_ubi_gray)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of Bukit_ubi_gray')
plt.plot(days,lstm_fc_ubi_gray[7:].flatten(),label='Bukit_ubi_gray',color='red')
plt.ylabel('Total Grey')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[52]:


X_train , y_train , X_test , y_test  = st.processing(panching_gray['Total Grey'])
####################################
model_lstm_panching_gray = lstm(X_train,y_train)
lstm_fc_panching_gray = forecast_lstm_cnn(X_test,model=model_lstm_panching_gray)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of Panching_gray')
plt.plot(days,lstm_fc_panching_gray[7:].flatten(),label='panching_gray',color='red')
plt.ylabel('Total Grey')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[53]:


X_train , y_train , X_test , y_test  = st.processing(lembing_gray['Total Grey'])
####################################
model_lstm_lembing_gray = lstm(X_train,y_train)
lstm_fc_lembing_gray = forecast_lstm_cnn(X_test,model=model_lstm_lembing_gray)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of sq_lembing_gray')
plt.plot(days,lstm_fc_lembing_gray[7:].flatten(),label='sq_lembing_gray',color='red')
plt.ylabel('Total Grey')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[54]:


X_train , y_train , X_test , y_test  = st.processing(semambu_gray['Total Grey'])
####################################
model_lstm_semambu_gray = lstm(X_train,y_train)
lstm_fc_semambu_gray = forecast_lstm_cnn(X_test,model=model_lstm_semambu_gray)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of semambu_gray')
plt.plot(days,lstm_fc_semambu_gray[7:].flatten(),label='semambu_gray',color='red')
plt.ylabel('Total Grey')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[56]:


X_train , y_train , X_test , y_test  = st.processing(sagu_blue['TOTAL BWF'])
####################################
model_lstm_Sagu_Blue = lstm(X_train,y_train)
lstm_fc_Sagu_Blue = forecast_lstm_cnn(X_test,model=model_lstm_Sagu_Blue)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of Bukit_Sagu_Blue')
plt.plot(days,lstm_fc_Sagu_Blue[7:].flatten(),label='Bukit_Sagu_Blue',color='red')
plt.ylabel('Total Blue')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[58]:


X_train , y_train , X_test , y_test  = st.processing(ubi_blue['TOTAL BWF'])
####################################
model_lstm_ubi_blue = lstm(X_train,y_train)
lstm_fc_ubi_blue = forecast_lstm_cnn(X_test,model=model_lstm_ubi_blue)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of Bukit_ubi_blue')
plt.plot(days,lstm_fc_ubi_blue[7:].flatten(),label='Bukit_ubi_blue',color='red')
plt.ylabel('Total Blue')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[59]:


X_train , y_train , X_test , y_test  = st.processing(panching_blue['TOTAL BWF'])
####################################
model_lstm_panching_blue = lstm(X_train,y_train)
lstm_fc_panching_blue = forecast_lstm_cnn(X_test,model=model_lstm_panching_blue)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of Panching_Blue')
plt.plot(days,lstm_fc_panching_blue[7:].flatten(),label='panching_blue',color='red')
plt.ylabel('Total Blue')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[60]:


X_train , y_train , X_test , y_test  = st.processing(lembing_blue['TOTAL BWF'])
####################################
model_lstm_lembing_blue = lstm(X_train,y_train)
lstm_fc_lembing_blue = forecast_lstm_cnn(X_test,model=model_lstm_lembing_blue)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of sq_lembing_blue')
plt.plot(days,lstm_fc_lembing_blue[7:].flatten(),label='sq_lembing_blue',color='red')
plt.ylabel('Total Blue')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# In[61]:


X_train , y_train , X_test , y_test  = st.processing(semambu_blue['TOTAL BWF'])
####################################
model_lstm_semambu_blue = lstm(X_train,y_train)
lstm_fc_semambu_blue = forecast_lstm_cnn(X_test,model=model_lstm_semambu_blue)
####################################
plt.figure(figsize=(16,4))
plt.title('Forecasting of semambu_blue')
plt.plot(days,lstm_fc_semambu_blue[7:].flatten(),label='semambu_blue',color='red')
plt.ylabel('Total Blue')
plt.xlabel('Furute Days')
plt.legend()
plt.show()


# # Possion Distribution

# In[62]:


def Poisson(x,lam):
    prob = []
    for i in x:
        top1 = lam**i
        top2 = np.e**(-lam)
        mult = top1*top2
        bottom = math.factorial(math.ceil(i))
        P = mult/bottom
        prob.append(P)
    return prob
    


# In[68]:


monthly_sum = sagu_gray['2020']['Total Grey'].resample('M').sum()
lam = monthly_sum.mean()
x = np.arange(55,120,5)
y = Poisson(x,lam=monthly_sum.mean())
############################################################
plt.figure(figsize=(18,6))
plt.title('The 2021 sagu_gray Probability Distribution of Total Grey in a Month')
sns.barplot(x, y,color='orange')
plt.ylabel('Probability')
plt.xlabel('Total Grey')
plt.show()


# In[69]:


monthly_sum = ubi_gray['2020']['Total Grey'].resample('M').sum()
lam = monthly_sum.mean()
x = np.arange(0,50,2)
y = Poisson(x,lam=monthly_sum.mean())
############################################################
plt.figure(figsize=(18,6))
plt.title('The 2021 ubi_gray Probability Distribution of Total Grey in a Month')
sns.barplot(x, y,color='red')
plt.ylabel('Probability')
plt.xlabel('Total Grey')
plt.show()


# In[70]:


monthly_sum = panching_gray['2020']['Total Grey'].resample('M').sum()
lam = monthly_sum.mean()
x = np.arange(0,50,2)
y = Poisson(x,lam=monthly_sum.mean())
############################################################
plt.figure(figsize=(18,6))
plt.title('The 2021 panching_gray Probability Distribution of Total Grey in a Month')
sns.barplot(x, y,color='seagreen')
plt.ylabel('Probability')
plt.xlabel('Total Grey')
plt.show()


# In[71]:


monthly_sum = lembing_gray['2020']['Total Grey'].resample('M').sum()
lam = monthly_sum.mean()
x = np.arange(20,100,5)
y = Poisson(x,lam=monthly_sum.mean())
############################################################
plt.figure(figsize=(18,6))
plt.title('The 2021 lembing_gray Probability Distribution of Total Grey in a Month')
sns.barplot(x, y,color='blue')
plt.ylabel('Probability')
plt.xlabel('Total Grey')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




