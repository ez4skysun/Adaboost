import datetime

import deepdish as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# from sklearn.metrics import calinski_harabasz_score


def get_data(site):
    dict = dd.io.load(site)
    df_list = []
    for key in dict.keys():
        temp = dict[key]
        temp_df = pd.DataFrame(temp['Factor'],index=np.arange(1,41),columns=temp['StockCode'][0]).T
        temp_df['return'] = temp['ReturnDaily'][0]
        temp_df['date'] = key
        df_list.append(temp_df.reset_index().rename(columns={'index':'stockcode'}))
    return pd.concat(df_list)

def process_data(data, kmeans=False):
    # kmeans：是否根据因子值进行kmeans聚类对收益率降噪
    # 每日前30%收益率标记为+1，每日后30%收益率标记为-1，舍弃中间40%的数据
    # kmeans的超参数n_clusters是基于长期数据根据calinski_harabasz_score进行挑选的
    data.date = data.date.apply(lambda x:datetime.datetime(int(x[:4]),int(x[4:6]),int(x[6:8])))
    # 此处取了2020.4以后的数据
    data = data.loc[data.date >= datetime.datetime(2020,4,11)]
    data.set_index('date',inplace=True)
    time_list = data.index.unique()
    if kmeans:
        data['cluster'] = np.nan
        for time in time_list:
            kmeans = KMeans(n_clusters=29)
            kmeans.fit(data.loc[data.index == time].dropna(axis=1))
            data.loc[data.index == time,'cluster'] = kmeans.labels_ + 1
            temp = data.loc[data.index == time]
            for c in range(1,11):
                temp.loc[temp.cluster == c,'return'] = temp.groupby('cluster').mean().loc[c,'return']
        data.pop('cluster')
    data['label'] = 0
    def do_label(x,inf,sup):
        if x >= sup:
            return 1
        elif x <= inf:
            return -1
        else:
            return 0
    for date in time_list:
        temp = data[data.index == date]
        inf = temp['return'].quantile(0.3)
        sup = temp['return'].quantile(0.7)
        data.loc[data.index == date,'label'] = temp['return'].apply(lambda x :do_label(x,inf,sup))
    return data.loc[data.label != 0,:]
