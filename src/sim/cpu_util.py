import numpy as np
import pandas as pd
from scipy.stats import norm


class CpuUtil(object):
	def __read_cpu_util(self, cpu_util_file, app):
		df = pd.DataFrame.from_csv(cpu_util_file, sep='\t', header=0)
		return df.loc[df['app'] == app]

	
	def __init__(self, cpu_util_file, app, mst_tru):
		# linearly interpolate mean and std, both arrays are 0-based		
		cpu_df = self.__read_cpu_util(cpu_util_file, app)
		self.m_max = max(cpu_df['m'])
		self.mst = mst_tru.mst_mean
		self.cpu_mean = np.interp(range(0, self.m_max + 1), cpu_df['m'], cpu_df['mean'])
		self.cpu_std = np.interp(range(0, self.m_max + 1), cpu_df['m'], cpu_df['std'])


	def sample(self, m, workload, backlog):
		if m <= 0 or self.m_max < m:
			return 0

		while True:
			rnd = norm.rvs(loc=self.cpu_mean[m], scale=self.cpu_std[m], size=1)[0]
			if (0 < rnd):
				break;

		# cpu util is proportinal to the processing load
		load_intensity = min((workload + backlog) / self.mst[m], 1.0)

		# at most 100%
		return min(load_intensity * rnd, 100.0)


