#!/usr/bin/env python
# coding: utf-8

# ## 1. Data & Librarys

# In[30]:


import numpy as np
import pandas as pd

#시각화
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib import rc 
get_ipython().run_line_magic('matplotlib', 'inline')
rc('font', family='malgun gothic')
import seaborn as sns

#전처리
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#time
from datetime import datetime, timedelta

#Time Series
import statsmodels as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.collections import PolyCollection
from statsmodels.tsa.stattools import adfuller


# In[2]:


data=pd.read_csv('data/data_inc_3.csv')
y=pd.read_csv('data/중앙시장_시간대별_주차수.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# ## 2.Preprocessing

# In[5]:


data['date']=pd.to_datetime(data.date)
train_data=data.iloc[:-360,:]
test_data=data.iloc[-360:,:]
y['date1']=pd.to_datetime(y.date1)
y_train=y.iloc[:-361,:]
y_test=y.iloc[-361:-1,:]


# In[6]:


scaler = MinMaxScaler()
train_scaled=scaler.fit_transform(train_data.iloc[:,1:])
test_scaled=scaler.transform(test_data.iloc[:,1:])
data_scaled=scaler.transform(data.iloc[:,1:])


# In[7]:


y['parked'][:500].plot()
plt.title('주차장에 주차되어 있는 수(대)')
plt.show()


# In[8]:


data_s=pd.DataFrame(data_scaled)
train_s=pd.DataFrame(train_scaled)
train_parked=y_train['parked']
test_parked=y_test['parked']
parked=y['parked'][:-1]


# In[9]:


print(train_parked.shape,test_parked.shape,parked.shape)


# **PCA**  \
# 시계열 데이터의 외부변수는 상관관계가 높지 않는 것이 좋기 때문에 PCA를 통해 상관관계가 0인 변수들을 사용

# In[16]:


pca=PCA().fit(train_s) 

plt.plot(np.arange(1,train_s.shape[1]+1), pca.explained_variance_ratio_,color='darkcyan')
plt.xlabel('Number of PCA')
plt.ylabel('Explained variance ratio')
plt.xlim(0,20)
plt.show()


# In[17]:


pca = PCA(n_components=3) 
train_pca= pca.fit_transform(train_s)
test_pca= pca.transform(test_scaled)
data_pca= pca.transform(data_scaled)


# In[18]:


train_pca=pd.DataFrame(train_pca)
test_pca=pd.DataFrame(test_pca)
data_pca=pd.DataFrame(data_pca)


# In[19]:


train_pca.shape


# In[20]:


train_pca


# **시간 변수 추가** \
# 시계열 데이터의 시계열적인 특성을 부각하기 위해 시간대별, 요일별 더미 변수, 시간대별 차분 변수를 추가해주었다.

# In[21]:


def timestamp(x,data2,a='day',b='hour'):
    ind=x.index.tolist()
    x=x.reset_index(drop=True)
    x['date']=pd.to_datetime(x.date)
    x[a]=x.date.dt.dayofweek
    x[b]=x.date.dt.hour
    
    def times(c):
        if c in [1,2,3]:
            return('새벽A')
        elif c in [4,5,6]:
            return('새벽B')
        elif c in [7,8,9]:
            return('아침A')
        elif c in [10,11,12]:
            return('아침B')
        elif c in [13,14,15]:
            return('오후A')
        elif c in [16,17,18]:
            return('오후B')
        elif c in [19,20,21]:
            return('저녁A')
        else:
            return('저녁B')
    
    x['시간대'] = x[b].apply(times)
    
    for i in [a,b,'시간대']:
        globals()['dummy_'+str(i)] = pd.get_dummies(x[i])
        if i ==a:
            globals()['dummy_'+str(i)].columns=['월','화','수','목','금','토','일']
        elif i==b:
            globals()['dummy_'+str(i)].columns=[str(x)+'시' for x in range(24)]
        data2=pd.concat([data2,globals()['dummy_'+str(i)]],axis=1)
    

    
    
    #display(data2['parked_shifted_48'])
    data2=data2.dropna()

    return data2

train_data2=timestamp(train_data,train_pca,'day','hour')
test_data2=timestamp(test_data,test_pca,'day','hour')
data2=timestamp(data,data_pca,'day','hour')


# In[22]:


df=pd.DataFrame(np.zeros(360))


# In[23]:


train_data22=train_data2.iloc[:,3:]


# In[24]:


test_data22=test_data2.iloc[:,3:]


# In[25]:


data22=data2.iloc[:,3:]


# In[26]:


start_date = datetime.strptime('2021-04-01 00:00', '%Y-%m-%d %H:%M') 
end_date = datetime.strptime('2021-04-16 00:00', '%Y-%m-%d %H:%M') 

# 날짜를 입력할 리스트 
str_date_list = [] 

while start_date.strftime('%Y-%m-%d %H:%M') != end_date.strftime('%Y-%m-%d %H:%M'): 
    str_date_list.append(start_date.strftime('%Y-%m-%d %H:%M')) 
    start_date += timedelta(hours=1) 


# In[27]:


df['date']=str_date_list


# In[28]:


submit_data=timestamp(data,df,'day','hour')


# In[31]:


sns.heatmap(train_data2.corr(),cmap='YlGn')
plt.show()


# In[41]:


data2.to_csv('data/dataplus2.csv',index=False)


# In[239]:


train_data2.to_csv('data/train_data2.csv',index=False)
test_data2.to_csv('data/test_data2.csv',index=False)


# ## 3. Time Series

# **시계열 분해**
# 

# In[10]:


decomposition = seasonal_decompose(train_parked, model='additive', period=24)

fig, axes = plt.subplots(4, 1, sharex=True)

decomposition.observed.plot(ax=axes[0], legend=False, color='darkcyan')
axes[0].set_ylabel('Observed')
decomposition.trend.plot(ax=axes[1], legend=False, color='darkcyan')
axes[1].set_ylabel('Trend')
decomposition.seasonal.plot(ax=axes[2], legend=False, color='darkcyan')
axes[2].set_ylabel('Seasonal')
decomposition.resid.plot(ax=axes[3], legend=False, color='darkcyan')
axes[3].set_ylabel('Residual')
plt.show()


# **정상성 검증(ADF검정)**

# In[11]:


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=120).mean()
    rolstd = timeseries.rolling( window=120).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(10, 6))
    orig = plt.plot(timeseries, color='darkcyan',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print ('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[12]:


test_stationarity(train_parked)


# **차분진행**

# In[13]:


#PACF와 ACF 그래프를 통해 적절한 차수 탐색

fig, axes = plt.subplots(1,2, figsize=(12,3))
plot_acf(train_parked,ax=axes[0],color='darkcyan',vlines_kwargs={"colors":'darkcyan'})
plot_pacf(train_parked, ax=axes[1],color='darkcyan',vlines_kwargs={"colors":'darkcyan'} )

for item in axes[0].collections:
    if type(item)==PolyCollection:
        item.set_facecolor('darkcyan')
        
for item in axes[1].collections:
    if type(item)==PolyCollection:
        item.set_facecolor('darkcyan')

plt.show()


# In[14]:


y_train['diff']=y_train['parked'].diff(24)
diffed=y_train['diff'].dropna()
y_test['diff']=y_test['parked'].diff(24)
test_diffed=y_test['diff'].dropna()


# In[15]:


#차분 후, 정상성이 있는 시계열로 변환하였다.

test_stationarity(diffed)


# **계절성 차분** \
# 48시간을 주기로 기댓값이 변화하는 모습이 보여 계절성 차분을 진행한다.

# In[22]:


fig, axes = plt.subplots(1,2, figsize=(12,3))
plot_acf(diffed,ax=axes[0],color='darkcyan',vlines_kwargs={"colors":'darkcyan'})
plot_pacf(diffed, ax=axes[1],color='darkcyan',vlines_kwargs={"colors":'darkcyan'} )

for item in axes[0].collections:
    if type(item)==PolyCollection:
        item.set_facecolor('darkcyan')
        
for item in axes[1].collections:
    if type(item)==PolyCollection:
        item.set_facecolor('darkcyan')
plt.show()


# In[23]:


#계절성 차분
y_train['seasonal_first_difference'] = y_train['diff'] - y_train['diff'].shift(48)  
y_test['seasonal_first_difference'] = y_test['diff'] - y_test['diff'].shift(48)  


# In[24]:


test_stationarity(y_train.seasonal_first_difference.dropna(inplace=False))


# In[25]:


season_diff=y_train.seasonal_first_difference.dropna(inplace=False)


# In[26]:


fig, axes = plt.subplots(1,2, figsize=(12,3))
plot_acf(season_diff,ax=axes[0],color='darkcyan',vlines_kwargs={"colors":'darkcyan'})
plot_pacf(season_diff, ax=axes[1],color='darkcyan',vlines_kwargs={"colors": 'darkcyan'})

for item in axes[0].collections:
    if type(item)==PolyCollection:
        item.set_facecolor('darkcyan')
        
for item in axes[1].collections:
    if type(item)==PolyCollection:
        item.set_facecolor('darkcyan')
plt.show()


# **로그변환**

# In[36]:


train_park=np.log(train_parked)


# In[37]:


test_park=np.log(test_parked)


# In[38]:


park = np.log(parked)


# ### ARIMA

# **Auto Arima** 
# - 차수를 조절하며 AIC를 최소로 하는 최적의 차수를 찾는 방법 
# - 평가지표: AUC

# In[39]:


from pmdarima.arima import auto_arima

model_arima= auto_arima(train_park,X=train_data22,trace=True, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,d=1,suppress_warnings=True,stepwise=False,seasonal=True)

model_arima.fit(parked)


# In[40]:


model_arima.plot_diagnostics(figsize=(10,10))


# **SARIMAX**
# - SARIMA: arima 모델에 계절성 성분이 추가된 모델로 데이터의 주기(M)의 정보를 활용
# - SARIMAX: 기존의 SARIMA모델에 외부 변수를 활용

# In[117]:


arimax = sm.tsa.statespace.sarimax.SARIMAX(train_park,order=(3,1,2),seasonal_order=(3,1,2,48),exog = train_data22,
                                  enforce_stationarity=False, enforce_invertibility=False,n_jobs=-1).fit()
arimax.summary()


# In[309]:


arimax.aic


# In[154]:


fcast_arima = arimax.predict(1800,1799+test_data2.shape[0],dynamic=False,exog=test_data22)


# In[140]:


fcast_arima1=np.exp(fcast_arima)


# In[310]:


test_parked.reset_index(drop=True).plot(color='crimson')
pd.Series(fcast_arima1).reset_index(drop=True).plot(color='darkcyan')
plt.legend(['target','predict'],bbox_to_anchor=(1,1))


# In[126]:


from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(test_parked, fcast_arima1)


# **Test**

# In[253]:


fcast_arima1.index=data['date'][1800:]


# In[256]:


train_parked.index=data['date'][:1800]


# In[316]:


parked.index=data['date']


# In[317]:


plt.figure(figsize=(15,7))
line1, = plt.plot(parked,  color='darkcyan')
#line2, = plt.plot(arimax.fittedvalues, marker='o', color='blue')
line3, = plt.plot(fcast_arima1,  color='crimson')
plt.legend([line1, line3],['original', 'ARIMA'],fontsize=15)
plt.show()


# **Ensemble** \
# ARIMA와 LSTM의 예측값 앙상블

# In[33]:


minmax=pd.read_csv('data/LSTM_0.042.csv')


# In[130]:


arim=fcast_arima1.reset_index(drop=True)


# In[131]:


lstm=minmax['y_pred']


# In[178]:


preds = [arim, lstm]


# In[190]:


p = 2.57
p = (np.sum(np.array(preds)**p, axis=0) / len(preds))**(1/p)


# In[193]:


p=arim*0.2+lstm*0.8


# In[186]:


from scipy.stats.mstats import gmean
p=gmean([arim,lstm])


# In[309]:


test_parked.reset_index(drop=True).plot(color='crimson')
pd.Series(p).reset_index(drop=True).plot(color='darkcyan')
plt.legend(['target','predict'],fontsize=12,bbox_to_anchor=(1,1))


# In[195]:


from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(test_parked, p)


# ## Predict
# **4월 이후 15일 예측**

# In[414]:


parked


# In[51]:


park=np.log(parked[48:])


# In[156]:


data2.shape


# In[163]:


final_arimax = sm.tsa.statespace.sarimax.SARIMAX(park,order=(2,1,3),seasonal_order=(2,1,3,48),exog = data22,
                                  enforce_stationarity=False, enforce_invertibility=False,).fit()
final_arimax.summary()


# In[162]:


data2


# In[165]:


final_arima = final_arimax.predict(2160,2519,dynamic=False,exog=submit_data.iloc[:,2:])


# In[166]:


final_arima=np.exp(final_arima)


# In[170]:


final_lstm=pd.read_csv('data/LSTM_April.csv')


# In[322]:


final_lstm.index=[i for i in range(2160,2520)]


# In[323]:


final_lstm.plot(color='darkgreen')


# In[321]:


final_arima.index=[i for i in range(2160,2520)]
final_arima.plot(color='darkgreen')
plt.show()


# In[173]:


final_preds = [final_arima, final_lstm]


# In[324]:


p=final_arima*0.2+final_lstm['prediction']*0.8


# In[325]:


p.plot(color='darkgreen')
plt.title('주차장의 4월 수요 예측')
plt.show()


# In[330]:


p.index=df['date']


# In[335]:


parked.index


# In[338]:


p.index=pd.to_datetime(p.index)


# In[340]:


plt.figure(figsize=(15,7))
line1, = plt.plot(parked,  color='darkcyan')
#line2, = plt.plot(arimax.fittedvalues, marker='o', color='blue')
line3, = plt.plot(p,  color='crimson')
plt.legend([line1, line3],['original', 'predict'],fontsize=15)
plt.show()


# In[ ]:




