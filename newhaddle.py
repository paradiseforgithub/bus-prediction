import pandas as pd
from functools import reduce
from itertools import groupby
import numpy as np
from multiprocessing import Pool
import os
import re
import time

# 命令行显示格式设置
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',5000)

# path = './../traindata'
path = 'E:\\DATA\\tianjinbus\\traindata'

namelist = list(filter(lambda x: x.split('.')[1]=='csv', os.listdir(path)))
# print(namelist)

def haddle_time():
    global namelist
    all = []
    pattern = re.compile(r'\d+')
    name = list(map(lambda x: x[0][:4]+'-'+x[0][4:6]+'-'+x[0][6:], list(map(lambda x :pattern.findall(x), namelist))))
    # print(name)
    # 为了显示进度
    process_N = len(namelist)
    process_i = 0
    st = time.clock()

    for i in range(len(namelist)):
        # print(name[i])
        p = round((process_i + 1) * 100 / process_N)
        duration = round(time.clock() - st, 2)
        remaining = round(duration * 100 / (0.01 + p) - duration, 2)
        print("进度:{0}%，已耗时:{1} s，预计剩余时间:{2}  s. ".format(p, duration, remaining), end="\r")
        process_i += 1

        df = pd.read_csv(path+'/'+namelist[i])
        df['O_TIME'] = df['O_TIME'].apply(lambda x: name[i]+' '+x)
        df['time_stamp'] = df['O_TIME'].map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        df['O_TIME'] = pd.to_datetime(df['O_TIME'])
        df['day'] = df['O_TIME'].dt.day
        df = df.drop(['sames'], axis=1)
        all.append(df)
    df_all = pd.concat(all, axis=0).sort_values(by='O_TIME').reset_index(drop=True)
    # print(df_all)
    df_all.to_csv('./../traindata.csv', index=None)
    return df_all

def station_time(df):
    df_all = []
    # 为了显示进度
    process_N = len(df.groupby(['day','O_TERMINALNO','O_UP']))
    process_i = 0
    st = time.clock()
    # 先按天，公交ID，上下行来分组操作，最后进行合并
    for x in df.groupby(['day','O_TERMINALNO','O_UP']):
        day = x[0][0]

        # 显示进度设置
        p = round((process_i + 1) * 100 / process_N)
        duration = round(time.clock() - st, 2)
        remaining = round(duration * 100 / (0.01 + p) - duration, 2)
        print("进度:{0}%，已耗时:{1} s，预计剩余时间:{2}  s. ".format(p, duration, remaining), end="\r")
        process_i += 1

        df_group = x[1].reset_index(drop=True)
        # diff()的参数-1表示逆着减
        df_group['sames_stamp'] = df_group['time_stamp'].diff(-1).map(lambda x: -x)

        df_group['sames_station'] = df_group['O_NEXTSTATIONNO'].diff(-1).map(lambda x: -x)
        # sames_station = df_group['O_NEXTSTATIONNO'].diff()
        max_stationno = df_group['O_NEXTSTATIONNO'].unique().max()
        # print(df_group)
        buses = []
        j = 0
        for i in df_group['sames_stamp'].index:
            # 我们认为大于60分钟的认为是两班车或者站数相差为负数的
            if df_group.loc[i, 'sames_stamp'] > 3600 or df_group.loc[i, 'sames_station'] < 0:
                temp = df_group.ix[j:i, ]
                temp['sames_stamp'] = temp['time_stamp'].diff(-1).map(lambda x: -x)
                # 去掉站点差值为0的数据，再相减
                temp['sames_station'] = temp['O_NEXTSTATIONNO'].diff()
                # 重置一下索引是为了删除idxForDel的时候索引(t+1)为空
                temp = temp[temp['sames_station']!=0].reset_index(drop=True)
                temp['sames_station'] = temp['O_NEXTSTATIONNO'].diff(-1).map(lambda x: -x)
                # 这个循环是为了去掉不连续的值
                idxForDel = []
                for t in temp.index:
                    # 如果O_NEXTSTATION为2并且在整点附近，就把到站时间加上与该整点的插值，即time_stamp%3600的值
                    if temp.loc[t, 'O_NEXTSTATIONNO'] == 2 and temp.loc[t,'time_stamp']%3600<=120:
                        temp.loc[t,'sames_stamp'] = temp.loc[t,'time_stamp']%3600+temp.loc[t,'sames_stamp']
                    # 去掉了值为大于1和值为NaN的情况, 同时去掉了到站间隔小于30秒的情况
                    if temp.loc[t, 'sames_station']!=1 or temp.loc[t, 'sames_stamp'] < 30:
                        idxForDel.append(t)
                # print(idxForDel)
                temp = temp.drop(idxForDel)
                buses.append(temp)
                j = i+1
        if len(buses)>0:
            df_all.append(pd.concat(buses, axis=0).reset_index(drop=True))
    pd.concat(df_all, axis=0).reset_index(drop=True).to_csv(day+'_newhaddletrain.csv', index=None)
    return pd.concat(df_all, axis=0).reset_index(drop=True)

def haddle_missing(df):
    # 跟据测试集的数据找出要预测的公交，将对应的公交车在训练集中提取出来，进行缺失值的填补处理
    df_test = pd.read_csv('E:\\DATA\\tianjinbus\\toBePredicted_forUser.csv')
    df_train = df
     l_ter = pd.unique(df_test['O_TERMINALNO'])
    def dic_max_stationno(df):
        dic = {}
        lineno = df['O_LINENO'].unique()
        for up in range(2): 
            for no in lineno:
                temp = df[(df['O_LINENO']==no)&(df['O_UP']==up)]['O_NEXTSTATIONNO'].unique()
                if len(temp)>0:
                    dic[(up, no)] = temp.max()
                else:
                    pass
        return dic
    # print(dic_max_stationno(df_train))
    d_max_stationno = dic_max_stationno(df_train)
    # print(d_max_stationno)
    ter_up_df_list = []
    for terminalno in l_ter:
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>', terminalno)
        up_state = pd.unique(df_test[df_test['O_TERMINALNO']==terminalno]['O_UP'])
        # pd_list是与df_join，join之后的df列表
        up_df_list = []
        for up in up_state:
            O_TERMINALNO = terminalno
            O_UP = up
            df = df_train[(df_train['O_TERMINALNO']==terminalno) & (df_train['O_UP']==up)].sort_values(by='O_TIME').reset_index(drop=True)
            if not df.empty:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>', terminalno)
                O_LINENO = df.loc[0, 'O_LINENO']
                # print(df.shape)
                df['sames_station'] = df['O_NEXTSTATIONNO'].diff(-1)
                # print(df)
                idx = []
                for i in df.index:
                    if df['sames_station'][i] > 0:
                        idx.append(i)
                # print(idx)
                # print(d_max_stationno.get((O_UP, O_LINENO)))
                if d_max_stationno.get((O_UP, O_LINENO)):
                    df_join = pd.DataFrame([[i] for i in range(2,d_max_stationno.get((O_UP, O_LINENO))+1)], columns=['O_NEXTSTATIONNO'])
                if idx:
                    for i in range(len(idx)):
                        if i==0:
                            left = df.iloc[:idx[0]+1]
                        elif i==len(idx)-1:
                            left = df.iloc[idx[i]+1:]
                        else:
                            left = df.iloc[idx[i]+1:idx[i+1]+1]
                else:
                    left = df
                temp = pd.merge(df_join, left, how='left', on=['O_NEXTSTATIONNO'])
                temp['O_LINENO'] = temp['O_LINENO'].fillna(O_LINENO)
                temp['O_TERMINALNO'] = temp['O_TERMINALNO'].fillna(O_TERMINALNO)
                temp['O_UP'] = temp['O_UP'].fillna(O_UP)
                for i in temp.index:
                    if temp.loc[i,'O_LONGITUDE'] == 0:
                        lo = temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['O_LONGITUDE'].median()
                        temp.loc[i,'O_LONGITUDE'] = lo
                    if temp.loc[i,'O_LATITUDE'] == 0:
                        la = temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['O_LATITUDE'].median()
                        temp.loc[i,'O_LATITUDE'] = la
                    if np.isnan(temp.loc[i,'sames_stamp']):
                        sub_time_i = temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['sames_stamp'].mean()
                        longitude =  temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['O_LONGITUDE'].median()
                        latitude = temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['O_LATITUDE'].median()
                        # print(sub_time_i) 
                        temp.loc[i,'sames_stamp'] = sub_time_i
                        temp.loc[i,'O_LONGITUDE'] = longitude
                        temp.loc[i,'O_LATITUDE'] = latitude
                temp = temp.drop(['sames_station'], axis=1)
                # print(temp)
                up_df_list.append(temp)
            else:
                print("{0} is not in train data.".format(O_TERMINALNO))
        if up_df_list:
            up_df = pd.concat(up_df_list, axis=0)
            ter_up_df_list.append(up_df)
    ter_up_df = pd.concat(ter_up_df_list, axis=0)
    print(ter_up_df.shape)
    ter_up_df.to_csv('fillallmissing.csv', index=None)
    return ter_up_df


if __name__ == '__main__':
    # # df = haddle_time()
    # df = pd.read_csv('traindata.csv')
    # l = [x[1] for x in df.groupby('O_UP')]
    # # print([x[1].shape for x in dff])
    # from concurrent import futures
    # with futures.ThreadPoolExecutor() as executor:
    #     for future in executor.map(station_time, l):
    #         print(future)
    # # station_time(df)


    df_test = pd.read_csv('E:\\DATA\\tianjinbus\\toBePredicted_forUser.csv')
    df_train = pd.read_csv('1newhaddle_df1.csv')
    l_ter = pd.unique(df_test['O_TERMINALNO'])
    def dic_max_stationno(df):
        dic = {}
        lineno = df['O_LINENO'].unique()
        for up in range(2): 
            for no in lineno:
                temp = df[(df['O_LINENO']==no)&(df['O_UP']==up)]['O_NEXTSTATIONNO'].unique()
                if len(temp)>0:
                    dic[(up, no)] = temp.max()
                else:
                    pass
        return dic
    # print(dic_max_stationno(df_train))
    d_max_stationno = dic_max_stationno(df_train)
    # print(d_max_stationno)
    ter_up_df_list = []
    for terminalno in l_ter:
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>', terminalno)
        up_state = pd.unique(df_test[df_test['O_TERMINALNO']==terminalno]['O_UP'])
        # pd_list是与df_join，join之后的df列表
        up_df_list = []
        for up in up_state:
            O_TERMINALNO = terminalno
            O_UP = up
            df = df_train[(df_train['O_TERMINALNO']==terminalno) & (df_train['O_UP']==up)].sort_values(by='O_TIME').reset_index(drop=True)
            if not df.empty:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>', terminalno)
                O_LINENO = df.loc[0, 'O_LINENO']
                # print(df.shape)
                df['sames_station'] = df['O_NEXTSTATIONNO'].diff(-1)
                # print(df)
                idx = []
                for i in df.index:
                    if df['sames_station'][i] > 0:
                        idx.append(i)
                # print(idx)
                # print(d_max_stationno.get((O_UP, O_LINENO)))
                if d_max_stationno.get((O_UP, O_LINENO)):
                    df_join = pd.DataFrame([[i] for i in range(2,d_max_stationno.get((O_UP, O_LINENO))+1)], columns=['O_NEXTSTATIONNO'])
                if idx:
                    for i in range(len(idx)):
                        if i==0:
                            left = df.iloc[:idx[0]+1]
                        elif i==len(idx)-1:
                            left = df.iloc[idx[i]+1:]
                        else:
                            left = df.iloc[idx[i]+1:idx[i+1]+1]
                else:
                    left = df
                temp = pd.merge(df_join, left, how='left', on=['O_NEXTSTATIONNO'])
                temp['O_LINENO'] = temp['O_LINENO'].fillna(O_LINENO)
                temp['O_TERMINALNO'] = temp['O_TERMINALNO'].fillna(O_TERMINALNO)
                temp['O_UP'] = temp['O_UP'].fillna(O_UP)
                for i in temp.index:
                    if temp.loc[i,'O_LONGITUDE'] == 0:
                        lo = temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['O_LONGITUDE'].median()
                        temp.loc[i,'O_LONGITUDE'] = lo
                    if temp.loc[i,'O_LATITUDE'] == 0:
                        la = temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['O_LATITUDE'].median()
                        temp.loc[i,'O_LATITUDE'] = la
                    if np.isnan(temp.loc[i,'sames_stamp']):
                        sub_time_i = temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['sames_stamp'].mean()
                        longitude =  temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['O_LONGITUDE'].median()
                        latitude = temp[(temp['O_TERMINALNO']==O_TERMINALNO)&(temp['O_NEXTSTATIONNO']==temp['O_NEXTSTATIONNO'][i])]['O_LATITUDE'].median()
                        # print(sub_time_i) 
                        temp.loc[i,'sames_stamp'] = sub_time_i
                        temp.loc[i,'O_LONGITUDE'] = longitude
                        temp.loc[i,'O_LATITUDE'] = latitude
                temp = temp.drop(['sames_station'], axis=1)

                print(temp)
                up_df_list.append(temp)
            else:
                print("{0} is not in train data.".format(O_TERMINALNO))
        if up_df_list:
            up_df = pd.concat(up_df_list, axis=0)
            ter_up_df_list.append(up_df)
    ter_up_df = pd.concat(ter_up_df_list, axis=0)
    print(ter_up_df.shape)

                
