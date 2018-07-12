import pandas as pd
from functools import reduce
from itertools import groupby
import numpy as np
from multiprocessing import Pool

# 命令行显示格式设置
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',5000)

class HaddleBusData:
	def __init__(self, path):
		self.path = path
		self.df = pd.read_csv(self.path)
	def _parseFile(self):
		df_list = []
		df = self.df
		# 去掉运行状态为0的值, 必须按时间进行排序
		df = df[df['O_RUN']==1].sort_values(by=['O_TIME'])
		# 按O_LINENO,O_UP分组，并将分组结果存在list里，便于用针对不同terminalno的_getNewDF方法处理
		for x in df.groupby(['O_LINENO','O_UP']):
			df_list.append(x[1])
		return df_list

	def _day2sec(self, day):
		x = day.split(':')
		return int(x[0])*60*60 + int(x[1])*60 + int(x[2])

    # 对dataframe进行处理，处理后，O_NEXTSTATION相差为1的可以让time_stamp直接相减得到用时
    # 不为1的值和空值可以不用管，仅仅为了能下一个值相减
	def _getNewDF(self, df_lineno):
		len_station = df_lineno.drop_duplicates(['O_NEXTSTATIONNO']).loc[:,'O_NEXTSTATIONNO'].max()
		list_all_by_terminalno = []
		df_lineno['time_stamp'] = df_lineno.apply(lambda x: self._day2sec(x['O_TIME']), axis=1)
		for df in df_lineno.groupby('O_TERMINALNO'):
			df_group = df[1].reset_index(drop=True)
			bus_time_sub = df_group['time_stamp'].diff()
			# 仅仅用于同一班车的判断
			station_sub_temp = df_group['O_NEXTSTATIONNO'].diff()
			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>', df[0])
			# 将同一天同一路线的不同车次提取出来
			buses = []
			j = 0
			for i in bus_time_sub.index:
				# 我们认为大于30分钟的认为是两班车或者站数相差为负数的
				if bus_time_sub[i] > 1800 or station_sub_temp[i] < 0:
					buses.append(df_group.ix[j:i-1, ])
					j = i
			buses.append(df_group.ix[j:, ])
			# 新处理的buses
			newbuses = []
			for bus in buses:
				# 最后一站的处理
				if bus.iloc[-1:]['O_NEXTSTATIONNO'].values == len_station:
					bus.iloc[-1:]['O_NEXTSTATIONNO'] = len_station + 1 
				# 删除相邻站数相差不为1的站及其它，为了计算时间能够直接相减
				bus['diff_station'] = bus['O_NEXTSTATIONNO'].diff()
				bus = bus[bus['diff_station']!=0].fillna({'diff_station':1.0}).reset_index()
				temp_delete_index = []
				for i in range(bus.shape[0]):
					if i == 0 or i == bus.shape[0]-1:
						if bus['diff_station'][i] != 1:
							temp_delete_index.append(i)
					else:
						if bus['diff_station'][i] != 1:
							temp_delete_index.append(i)
						elif bus['diff_station'][i-1] != 1 and bus['diff_station'][i+1] != 1:
							temp_delete_index.append(i)
						else:
							pass
				bus = bus.drop(temp_delete_index)
				# 去掉最后一个值和第一个值不连续的情况
				bus['diff_station'] = bus['O_NEXTSTATIONNO'].diff()
				if bus.iloc[-1:]['diff_station'].values != 1:
					bus = bus.drop([bus.iloc[-1:].index.values[0]])
				if bus.iloc[1:2]['diff_station'].values != 1:
					bus = bus.drop([bus.iloc[:1].index.values[0]])
				bus['diff_station'] = bus['O_NEXTSTATIONNO'].diff()
				bus['sub_time_stamp'] = bus['time_stamp'].diff()
				bus = bus.reset_index(drop=True)
				# 区分每个bus里不连续的情况
				p = 0
				for q in bus.index:
					if bus['diff_station'][q] > 1.0:
						bus.loc[p:q-1, ]['diff_station'] = bus.loc[p:q-1, ]['O_NEXTSTATIONNO'].diff()
						bus.loc[p:q-1, ]['sub_time_stamp'] = bus.loc[p:q-1, ]['time_stamp'].diff()
						# print(bus.loc[p:q-1, ])
						p = q
				bus.loc[p:, ]['diff_station'] = bus.loc[p:, ]['O_NEXTSTATIONNO'].diff()
				bus.loc[p:, ]['sub_time_stamp'] = bus.loc[p:, ]['time_stamp'].diff()
				# print(bus.loc[p:, ])
				# print(bus)
				newbuses.append(bus)
			# 去掉长度小于5的dataframe
			newbuses = list(filter(lambda x: x.shape[0] > 5, newbuses))
			# 合并同一terminalno的每一班次公交车的数据
			if len(newbuses) > 0:
				new_df = pd.concat([x for x in newbuses], axis=0).reset_index(drop=True)
				list_all_by_terminalno.append(new_df)
		print(len(list_all_by_terminalno))
		if len(list_all_by_terminalno) > 0:
			all_by_terminalno = pd.concat([x for x in list_all_by_terminalno], axis=0).reset_index(drop=True) 
			# 删除时间间隔小于1分钟的数据
			all_by_terminalno = all_by_terminalno[(all_by_terminalno['sub_time_stamp'] > 60)]
			# 新建到站列，为O_NEXTSTATIONNO-1,并删除无用的列
			all_by_terminalno['stationno'] = all_by_terminalno['O_NEXTSTATIONNO'].apply(lambda x:x-1)
			all_by_terminalno = all_by_terminalno.drop(['index','O_NEXTSTATIONNO'], axis=1)
			return all_by_terminalno

	def combine(self):
		dfs = self._parseFile()
		temp = []
		for df in dfs:
			df = df.drop(['O_MIDDOOR','O_REARDOOR','O_FRONTDOOR'],axis=1)
			x = self._getNewDF(df)
			if x is not None:
				temp.append(x)
			print('==================================')
		result = pd.concat([x for x in temp], axis=0).reset_index(drop=True)
		return result

	def toCSV(self, df):
		filename = 'new_' + self.path
		df.to_csv(filename, index=None)
		print('Write CSV File OK!')

	# 获取所有车辆编号
	def getTerminalNO(self):
		return list(self.df.groupby('O_LINENO').size().index)

	def getDFByLineNO(self, terminalno):
		pass

if __name__ == '__main__':
	file_path1 = 'train20171001.csv'
	file_path2 = 'train20171002.csv'
	file_path3 = 'train20171003.csv'
	file_path4 = 'train20171004.csv'
	file_path5 = 'train20171005.csv'
	file_path6 = 'train20171006.csv'
	file_path7 = 'train20171007.csv'
	file_path8 = 'train20171008.csv'
	# file_path1 = 'lineno_1_all.csv'
	# file_path2 = 'new_lineno_1_all.csv't
	# file_path2 = 'new_train20171001.csv'
	# e1 = HaddleBusData(file_path1)
	# df_all = e1.combine()
	# print(len(df_all))
	# e1.toCSV(df_all)

	e1 = HaddleBusData(file_path1)
	e2 = HaddleBusData(file_path2)
	e3 = HaddleBusData(file_path3)
	e4 = HaddleBusData(file_path4)

	# e1.toCSV(e1.combine())

	p = Pool()
	elist = [e1.toCSV(e1.combine()),e2.toCSV(e2.combine()),e3.toCSV(e3.combine()),e4.toCSV(e4.combine())]
	for e in elist:
		p.apply_async(e)
	p.close()
	p.join()
	