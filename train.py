import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaboost import *
from utils import *


def daily_model_train(X,y):
    # 一天数据训练出一个模型
    a = X.index.nunique()
    def subjob_func(i):
        temp = X.index.unique()[i]
        X_temp = X.loc[X.index==temp]
        y_temp = y.loc[y.index==temp]
        x_train,x_val,y_train,y_val = train_test_split(X_temp,y_temp,test_size=0.1,random_state=i)
        model = DtcAdaboost(max_depth=2,min_samples_split=0.1,learning_rate=.3,k=0.025,n_estimators=100,early_stop=True)
        model.fit(x_train,y_train[['label']],y_train['return'],x_val,y_val)
        return model
    models = Parallel(n_jobs=5)(delayed(subjob_func)(i) for i in range(a))
    return models

def daily_factor_caculate(models,X,y,d=60):
    # 利用每天的一个模型进行前60天的60个模型预测结果指数加权
    ans = y.copy(deep=True)
    a = X.index.nunique()
    for i in range(0,a-d):
        time = X.index.unique()[i+d]
        prob_list = []
        for j in range(d):
            prob_list.append(models[i+j].predict_proba(X.loc[X.index==time]))
        w = [0.95**i for i in range(d-1,-1,-1)]
        ans.loc[y.index==time,'prob'] = np.dot(np.matrix(prob_list).T,np.array(w)).tolist()[0]
    return ans

def monthly_model_caculate(X,Y,d=60):
    # 利用前60天数据训练出一个模型预测下一天
    y = Y.copy(deep=True)
    a = X.index.nunique()
    def subjob_func(i):
        time_start = X.index.unique()[i]
        time_end = X.index.unique()[i+d-1]
        time = X.index.unique()[i+d]
        X_temp = X.loc[(X.index>=time_start)&(X.index<=time_end)]
        y_temp = y.loc[(y.index>=time_start)&(y.index<=time_end)]
        x_train,x_val,y_train,y_val = train_test_split(X_temp,y_temp,test_size=0.1,random_state=i)
        model = DtcAdaboost(max_depth=2,min_samples_split=0.1,learning_rate=.1,k=.0105,n_estimators=200,early_stop=True)
        model.fit(x_train,y_train[['label']],y_train['return'],x_val,y_val)
        return model.predict_proba(X.loc[X.index==time])
    res = Parallel(n_jobs=5)(delayed(subjob_func)(i) for i in range(0,a-d))
    for i in range(0,a-d):
        time = X.index.unique()[i+d]
        y.loc[X.index==time,'prob'] = res[i]
    return y

def monthly_model_train(X,y,d=20):
    # 利用每20天数据训练出一个模型
    a = X.index.nunique()
    def subjob_func(i):
        time_start = X.index.unique()[i]
        time_end = X.index.unique()[i+d-1]
        X_temp = X.loc[(X.index>=time_start)&(X.index<=time_end)]
        y_temp = y.loc[(y.index>=time_start)&(y.index<=time_end)]
        x_train,x_val,y_train,y_val = train_test_split(X_temp,y_temp,test_size=0.1,random_state=i)
        model = DtcAdaboost(max_depth=2,min_samples_split=0.1,learning_rate=.1,k=.0105,n_estimators=200,early_stop=True)
        model.fit(x_train,y_train[['label']],y_train['return'],x_val,y_val)
        return model
    res = Parallel(n_jobs=5)(delayed(subjob_func)(i) for i in range(0,a-d))
    return res

def monthly_factor_caculate(models,X,Y,td=20,tn=3):
    # 利用每20天数据训练出的模型,前60天的三个模型预测结果平均得出结果
    y = Y.copy(deep=True)
    a = X.index.nunique()
    for i in range(0,a-tn*td):
        time = X.index.unique()[i+tn*td]
        prob_list = []
        for j in range(tn):
            prob_list.append(models[i+j].predict_proba(X.loc[X.index==time]))
        # w = [0.95**i for i in range(tn-1,-1,-1)]
        w = [1/tn for _ in range(tn)]
        y.loc[X.index==time,'prob'] = np.dot(np.matrix(prob_list).T,np.array(w)).tolist()[0]
    return y

def main():
    data = get_data('D:\Project\lianhai\FactorData.h5')
    processed_data = process_data(data)
    X = processed_data[np.arange(1,41)]
    y = processed_data[['stockcode','label','return']]
    y['pred'] = np.nan
    y['prob'] = np.nan

    daily_models = daily_model_train(X,y)
    y1 = daily_factor_caculate(daily_models,y)

    y2 = monthly_model_caculate(X,y)

    monthly_models = monthly_model_train(X,y)
    y3 = monthly_factor_caculate(monthly_models,y)

    # joblib.dump(models,'models.dat') 
    # models = joblib.load('models.dat')

    # y1:每日模型60天加权因子 y2:六十天模型因子 y3:每二十天模型3模型平均因子

    y_test_1 = y1.loc[~y1.prob.isna(),:].copy(deep=True)
    y_test_2 = y2.loc[~y2.prob.isna(),:].copy(deep=True)
    y_test_3 = y3.loc[~y3.prob.isna(),:].copy(deep=True)
    # 9.54源于加权时的指数衰减序列
    y_test_1['pred'] = y_test_1.prob.apply(lambda x:-1 if x <= 9.54 else 1)
    y_test_2['pred'] = y_test_2.prob.apply(lambda x:-1 if x <= 0.5 else 1)
    y_test_3['pred'] = y_test_3.prob.apply(lambda x:-1 if x <= 0.5 else 1)
    y_test_1.to_csv('daily_factor_caculate')
    y_test_2.to_csv('monthly_model_caculate')
    y_test_3.to_csv('monthly_factor_caculate')
    return y_test_1, y_test_2, y_test_3


if __name__ == '__main__':

    y_test_1, y_test_2, y_test_3 = main()
    print('daily_factor_caculate IC :',y_test_1[['return','prob']].corr(method='spearman').iloc[0,1])
    print('monthly_model_caculate IC :',y_test_2[['return','prob']].corr(method='spearman').iloc[0,1])
    print('monthly_factor_caculate IC :',y_test_3[['return','prob']].corr(method='spearman').iloc[0,1])

    print('daily_factor_caculate confusion matirx :',confusion_matrix(y_true=y_test_1.label,y_pred=y_test_1.pred))
    print('monthly_model_caculate confusion matirx :',confusion_matrix(y_true=y_test_2.label,y_pred=y_test_2.pred))
    print('monthly_factor_caculate confusion matirx :',confusion_matrix(y_true=y_test_3.label,y_pred=y_test_3.pred))
    
    plt.plot(y_test_1.resample('BM')[['return','prob']].corr(method='spearman').unstack()['return']['prob'],label='daily_factor_caculate')
    plt.plot(y_test_2.resample('BM')[['return','prob']].corr(method='spearman').unstack()['return']['prob'],label='monthly_model_caculate')
    plt.plot(y_test_3.resample('BM')[['return','prob']].corr(method='spearman').unstack()['return']['prob'],label='monthly_factor_caculate')
    plt.legend()
    plt.show()
