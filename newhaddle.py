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

# path = '/Users/honey/Desktop/tianjinbus/data'
path = 'E:\\DATA\\tianjinbus\\traindata'

namelist = list(filter(lambda x: x.split('.')[1]=='csv', os.listdir(path)))
# print(namelist)

def haddle_time():
	global namelist
	all = []
	pattern = re.compile(r'\d+')
	name = list(map(lambda x: x[0][:4]+'-'+x[0][4:6]+'-'+x[0][6:], list(map(lambda x :pattern.findall(x), namelist))))
	# print(name)
	for i in range(1):
		print(name[i])
		df = pd.read_csv(path+'/'+namelist[i])
		df['O_TIME'] = df['O_TIME'].apply(lambda x: name[i]+' '+x)
		df['time_stamp'] = df['O_TIME'].map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
		df['O_TIME'] = pd.to_datetime(df['O_TIME'])
		df['day'] = df['O_TIME'].dt.day
		df = df.drop(['sames'], axis=1)
		all.append(df)
	df_all = pd.concat(all, axis=0).sort_values(by='O_TIME').reset_index(drop=True)
	# print(df_all)
	df_all.to_csv('traindata.csv', index=None)
	return df_all

def station_time(df):
	df_all = []
	# 先按天，公交ID，上下行来分组操作，最后进行合并
	for x in df.groupby(['day','O_TERMINALNO','O_UP']):
		print(x[0])
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
		df_all.append(pd.concat(buses, axis=0).reset_index(drop=True))
	return pd.concat(df_all, axis=0).reset_index(drop=True)


def haddle_missing(df):
	# 跟据测试集的数据找出要预测的公交，将对应的公交车在训练集中提取出来，进行缺失值的填补处理
	df_test = pd.read_csv('E:\\DATA\\tianjinbus\\toBePredicted_forUser.csv')


if __name__ == '__main__':
	# haddle_time()
	df_all = []
	df = pd.read_csv('traindata.csv')
	for x in df.groupby(['day','O_TERMINALNO','O_UP']):
		print(x[0])
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
		# print(len(buses))
		if len(buses)>0:
			df_all.append(pd.concat(buses, axis=0).reset_index(drop=True))
	pd.concat(df_all, axis=0).reset_index(drop=True).to_csv('newhaddle_df1.csv', index=None)

