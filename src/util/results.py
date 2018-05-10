import logging
import pandas as pd
import numpy as np
from core.scalr import ScalrState


log = logging.getLogger()


class Results(object):

	def __init__(self, sched_policy, app):
		self.df = pd.DataFrame(columns=['t', 'state', 'm', 'workload', 'mst_tru', 'backlog', 'cpu_util'])
		self.sched_policy = sched_policy
		self.app = app


	def add(self, datetime, t, state, m, workload, mst_tru, backlog, cpu_util):
		self.df.loc[datetime] = [t, state, m, workload, mst_tru, backlog, cpu_util];

		
	def print_stats(self):
		# violations = [1.0 if row['backlog'] <= self.target_backlog else 0 for index, row in self.df.iterrows()]
		ready_df = self.df[self.df.state == ScalrState.READY]
		satisfactions = [1.0 if row['backlog'] == 0 else 0 for index, row in ready_df.iterrows()]
		
		log.info('Stats: %s, %s, %.3f, %.3f, %.3f, %.3f' %
				 (self.sched_policy, self.app,
				  np.mean(self.df.m), np.mean(self.df.backlog),
				  100*np.mean(satisfactions), np.mean(ready_df.backlog)))


	def write_results(self, file):
		self.df.to_csv(file, sep='\t')
	

		
