import pandas as pd
from functools import reduce
from itertools import groupby
import numpy as np
import time

# 命令行显示格式设置
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',5000)

file_path = '901530_0.csv'
df = pd.read_csv(file_path)

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
# print(df.head(10))

idx = []
for i in df.index:
	if df['diff_time_stamp'][i] > 1800:
		idx.append(i)
# print(df.iloc[:idx[0]])
# print(df.dtypes)
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
# print(pd_list)
all = pd.concat([x for x in pd_list], axis=0).reset_index(drop=True)
# print(all)

for i in all.index:
	if np.isnan(all.iloc[i,7]):
		sub_time_i = all[all['stationno']==all['stationno'][i]]['sub_time_stamp'].mean()
		# print(sub_time_i) 
		all.iloc[i,7] = sub_time_i
print(all)

j=0
while true:
	np.isnan(all.iloc[j,8]):
		j += 1
	print(i, ' ', j)
