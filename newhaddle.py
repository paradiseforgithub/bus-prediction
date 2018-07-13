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

path = '/Users/honey/Desktop/tianjinbus/data'

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
	# 先按天，公交ID，上下行来分组操作，最后进行合并
	pass
	# df_list.append(x[1])


def haddle_missing(df):
	pass


if __name__ == '__main__':
	# haddle_time()
	df = pd.read_csv('traindata.csv')
	for x in df.groupby(['day','O_TERMINALNO','O_UP']):
		df_group = x[1].reset_index(drop=True)
		sames_stamp = df_group['time_stamp'].diff()
		sames_station = df_group['O_NEXTSTATIONNO'].diff()
		max_stationno = df_group['O_NEXTSTATIONNO'].unique().max()
		buses = []
		j = 0
		for i in sames_stamp.index:
			# 我们认为大于30分钟的认为是两班车或者站数相差为负数的
			if sames_stamp[i] > 1800 or sames_station[i] < 0:
				buses.append(df_group.ix[j:i-1, ])
				j = i
		buses.append(df_group.ix[j:, ])
		# print(len(buses))


