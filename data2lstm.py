import pandas as pd
from functools import reduce
from itertools import groupby
import numpy as np
import time

# 命令行显示格式设置
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',2000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',5000)
# 取消科学计数
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

# 需要提取的信息
O_LINENO = 1
O_TERMINALNO = 901530
O_UP = 0

df_train_all = pd.read_csv('new_train1-31.csv')
df_test = pd.read_csv('toBePredicted_forUser.csv')
# print(df_test.groupby(['O_LINENO','O_TERMINALNO','O_UP']).size())
print(df_test.shape)
l_ter = pd.unique(df_test['O_TERMINALNO'])
print(l_ter.shape)
for terminalno in l_ter:
	print('>>>>>>>>>>>>>>>>>>>>>>>>>', terminalno)
	up_state = pd.unique(df_test['O_UP'])
	for up in up_state:
		# 固定值填充
		# O_LINENO = 1
		O_TERMINALNO = terminalno
		O_UP = up
		df = df_train_all[(df_train_all['O_TERMINALNO']==terminalno) & (df_train_all['O_UP']==up)].sort_values(by='O_TIME').reset_index(drop=True)
		print(df.shape)

		# 时间戳转换由“20171001 07:07:23”转换成秒
		def fun_time_to_time(t):
			day = t.split(' ')[0]
			# moment = t.split(' ')[1]
			day = day[:4]+'-'+day[4:6]+'-'+day[6:]
			temp = day+' '+t.split(' ')[1]
			# x = time.strptime(temp, '%Y-%m-%d %H:%M:%S')
			# y = time.mktime(x)
			return temp

		# print(fun_time_to_time('20171001 07:07:43'))

		df['time_stamp'] = df['O_TIME'].map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
		df['diff_stationo'] = df['stationno'].diff()
		df['diff_time_stamp'] = df['time_stamp'].diff()

		idx = []
		for i in df.index:
			if df['diff_time_stamp'][i] > 1800 or df['diff_stationo'][i] < 0:
				idx.append(i)
		pd_list = []
		df_join = pd.DataFrame([[i] for i in range(2,32)], columns=['stationno'])
		for i in range(len(idx)):
			if i==0:
				left = df.iloc[:idx[0]]
			elif i==len(idx)-1:
				left = df.iloc[idx[i]:]
			else:
				left = df.iloc[idx[i]:idx[i+1]]
			# 如果每一趟公交车的有效记录小于总站数的一半，去掉
			if left.shape[0] > 15:
				pd_list.append(pd.merge(df_join, left, how='left', on=['stationno']))
		all = pd.concat([x for x in pd_list], axis=0).reset_index(drop=True)


		all['O_LINENO'] = all['O_LINENO'].fillna(O_LINENO)
		all['O_TERMINALNO'] = all['O_TERMINALNO'].fillna(O_TERMINALNO)
		all['O_UP'] = all['O_UP'].fillna(O_UP)

		for i in all.index:
			if all.iloc[i,4] == 0:
				lo = all[all['stationno']==all['stationno'][i]]['O_LONGITUDE'].median()
				all.iloc[i,4] = lo
			if all.iloc[i,5] == 0:
				la = all[all['stationno']==all['stationno'][i]]['O_LATITUDE'].median()
				all.iloc[i,5] = la
			if np.isnan(all.iloc[i,7]):
				sub_time_i = all[all['stationno']==all['stationno'][i]]['sub_time_stamp'].mean()
				longitude =  all[all['stationno']==all['stationno'][i]]['O_LONGITUDE'].median()
				latitude = all[all['stationno']==all['stationno'][i]]['O_LATITUDE'].median()
				# print(sub_time_i) 
				all.iloc[i,7] = sub_time_i
				all.iloc[i,4] = longitude
				all.iloc[i,5] = latitude
		all = all.drop(['diff_stationo', 'diff_time_stamp'], axis=1)
		# print(all)
		for i in range(int(all.shape[0]/30)):
			# 最后一个公交车站为空值的填补处理，从后往前
			if np.isnan(all.iloc[i*30+29, 8]):
				# temp为空值标记
				temp = 28
				while np.isnan(all.iloc[i*30+temp, 8]):
					temp -= 1
				# print(i*30+temp)
				base = all.iloc[i*30+temp, 8]
				for x in range(temp+1, 30):
					all.iloc[i*30+x, 8] = base + all.iloc[i*30+x, 7]
					base = all.iloc[i*30+x, 8]
			# 最后一个公交站非缺失值的处理，从前往后
			j = 0
			while j<30:
				if np.isnan(all.iloc[i*30+j, 8]):
					# temp_1为空值标记
					temp_1 = j
					while np.isnan(all.iloc[i*30+temp_1, 8]):
						temp_1 += 1
					# temp_len记录空值长度
					temp_len = temp_1 - j
					j = temp_1
					# print(i*30+j, '>>>>', temp_len)
					base_1 = all.iloc[i*30+j, 8]
					for x in range(temp_len):
						all.iloc[i*30+j-x-1, 8] = base_1 - all.iloc[i*30+j-x, 7]
						base_1 = all.iloc[i*30+j-x-1, 8]
				else:
					j += 1

		# 时间戳转化成时间
		all['O_TIME'] = pd.to_datetime(all['time_stamp'].map(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x))), format='%Y-%m-%d %H:%M:%S')
		all.sort_values(by='O_TIME', inplace=True)
		all = all.reset_index(drop=True)

		# 进行LSTM的数据预处理
		all['weekday'] = all['O_TIME'].dt.weekday
		all['hour'] = all['O_TIME'].dt.hour
		lstm_data = all.drop(['O_LINENO','O_TERMINALNO', 'O_TIME','O_LONGITUDE','O_LATITUDE', 'O_UP', 'time_stamp'], axis=1)

		# all.to_csv('new_901530_0.csv', index=None)
		lstm_data.to_csv('lstm_901530_0.csv', index=None)