import pandas as pd
import numpy as np

class Results(object):

	def __init__(self):
		self.df = pd.DataFrame(columns=['m', 'workload', 'mst_tru', 'backlog'])


	def add(self, datetime, m, workload, mst_tru, backlog):
		self.df.loc[datetime] = [m, workload, mst_tru, backlog];

		
	def print_stats(self):
		violations = [1.0 if row['workload'] > row['mst_tru'] else 0 for index, row in self.df.iterrows()]
		print('Stats: %.3f, %.3f, %.3f, %.3f' % 
			  (100 * np.mean(violations), 
			   np.mean(self.df['m']), 
			   np.mean(self.df['backlog']),
			   (np.mean(self.df['backlog']) / np.mean(self.df['workload']))))


	def write_results(self, file):
		self.df.to_csv(file, sep='\t')
	

		
