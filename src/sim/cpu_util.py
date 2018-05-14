import logging
import numpy as np
import pandas as pd
from scipy.stats import norm, dweibull

log = logging.getLogger()


class CpuUtil(object):
	def __read_cpu_util(self, cpu_util_file, app):
		df = pd.DataFrame.from_csv(cpu_util_file, sep='\t', header=0)
		return df.loc[df['app'] == app]

	
	def __init__(self, conf, mst_tru):
		self.mst = mst_tru.mst_mean
		self.conf = conf		
		self.cpu_util_dist = conf['cpu_util_dist']		

		# linearly interpolate mean and std, both arrays are 0-based
		cpu_df = self.__read_cpu_util(conf['cpu_util_files'][self.cpu_util_dist], conf['app'])
		self.m_max = max(cpu_df['m'])		
		if self.cpu_util_dist == 'norm':
			self.loc = np.interp(range(0, self.m_max + 1), cpu_df['m'], cpu_df['mean'])
			self.scale = np.interp(range(0, self.m_max + 1), cpu_df['m'], cpu_df['std'])
		elif self.cpu_util_dist == 'dweibull':
			self.c = np.interp(range(0, self.m_max + 1), cpu_df['m'], cpu_df['c'])
			self.loc = np.interp(range(0, self.m_max + 1), cpu_df['m'], cpu_df['loc'])
			self.scale = np.interp(range(0, self.m_max + 1), cpu_df['m'], cpu_df['scale'])
		else:
			log.error('Undefined cpu util dist: {}'.format(self.cpu_util_dist))


	def sample(self, m, workload, backlog):
		if m <= 0 or self.m_max < m:
			return 0

		while True:
			if self.cpu_util_dist == 'norm':			
				rnd = norm.rvs(loc=self.loc[m], scale=self.scale[m], size=1)[0]
			elif self.cpu_util_dist == 'dweibull':
				rnd = dweibull.rvs(self.c[m], loc=self.loc[m], scale=self.scale[m])
			else:
				log.error('Undefined cpu util dist: {}'.format(self.cpu_util_dist))
			if (0 < rnd):
				break;
		rnd = min(rnd, 100.0)

		# cpu util is proportinal to the processing load
		load_intensity = min((workload + backlog / self.conf['timestep_sec'] ) / self.mst[m], 1.0)

		# at most 100%
		return min(load_intensity * rnd, 100.0)


