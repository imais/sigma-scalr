import numpy as np
from scipy.stats import norm


class MstTru(object):
	def __init__(self, mst_df, random_sampling=True):
		# compute mean over all samples
		mst_mean = mst_df.iloc[:, 2:].mean(axis=1) 
		mst_std = mst_df.iloc[:, 2:].std(axis=1)
		self.m_max = max(mst_df['m'])

		# linearly interpolate mean and std, both arrays are 0-based
		self.mst_mean = np.interp(range(0, self.m_max + 1), mst_df['m'], mst_mean)
		self.mst_std = np.interp(range(0, self.m_max + 1), mst_df['m'], mst_std)

		self.random_sampling = random_sampling


	def sample(self, m):
		if m < 0 or self.m_max < m:
			return 0
		elif self.random_sampling:
			while True:
				rnd = norm.rvs(loc=self.mst_mean[m - 1], scale=self.mst_std[m - 1], size=1)
				if (0 < rnd):
					break;
			return rnd
		else:
			return self.mst_mean[m - 1] 

