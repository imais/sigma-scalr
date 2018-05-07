import logging
import pandas as pd
import numpy as np
from core.scalr import ScalrState


log = logging.getLogger()


class Results(object):

	def __init__(self, id=''):
		self.df = pd.DataFrame(columns=['t', 'state', 'm', 'workload', 'mst_tru', 'backlog'])
		self.id = id


	def add(self, datetime, t, state, m, workload, mst_tru, backlog):
		self.df.loc[datetime] = [t, state, m, workload, mst_tru, backlog];

		
	def print_stats(self):
		# violations = [1.0 if row['backlog'] <= self.target_backlog else 0 for index, row in self.df.iterrows()]
		ready_df = self.df[self.df.state == ScalrState.READY]
		satisfactions = [1.0 if row['backlog'] == 0 else 0 for index, row in ready_df.iterrows()]
		
		log.info('Stats(%s): %.3f, %.3f, %.3f, %.3f' %
				 (self.id, np.mean(self.df.m), np.mean(self.df.backlog),
				  100*np.mean(satisfactions), np.mean(ready_df.backlog)))


	def write_results(self, file):
		self.df.to_csv(file, sep='\t')
	

		
