import numpy as np
import pandas as pd
from scipy.stats import norm


class MstTru(object):
	def __read_mst_data(self, mst_data_file, app):
         df = pd.DataFrame.from_csv(mst_data_file, sep='\t', header=0)
         return df.loc[df['app'] == app]

	def __init__(self, mst_data_file, app):
		mst_df = self.__read_mst_data(mst_data_file, app)

		# compute mean over all samples
		mst_mean = mst_df.iloc[:, 2:].mean(axis=1) 
		mst_std = mst_df.iloc[:, 2:].std(axis=1)
		self.m_max = max(mst_df['m'])

		# linearly interpolate mean and std, both arrays are 0-based
		self.mst_mean = np.interp(range(0, self.m_max + 1), mst_df['m'], mst_mean)
		self.mst_std = np.interp(range(0, self.m_max + 1), mst_df['m'], mst_std)


	def sample(self, m):
		if m <= 0 or self.m_max < m:
			return 0

		# return self.mst_mean[m - 1]
		while True:
			rnd = norm.rvs(loc=self.mst_mean[m], scale=self.mst_std[m], size=1)[0]
			if (0 < rnd):
				break;

		return rnd


