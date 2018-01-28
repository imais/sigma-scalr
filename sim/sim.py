# from __future__ import absolute_import

import logging
import pandas as pd
from util.results import Results
from core.scalr import Scalr, ScalrState
from mst_tru import MstTru


log = logging.getLogger(__name__)


class Sim(object):
	def read_mst_data(self, mst_data_file, app):
         df = pd.DataFrame.from_csv(mst_data_file, sep='\t', header=0)
         return df.loc[df['app'] == app]


	def read_workload(self, workload_file, norm_factor):
		workload = pd.Series.from_csv(workload_file, sep='\t', header=0)
		if norm_factor < 0:
			return workload
		else:
			scale = norm_factor / workload.max()
			return workload.multiply(scale)		


	def __init__(self, conf):
		mst_df = self.read_mst_data(conf['mst_data_file'], conf['app'])
		self.scalr = Scalr(conf, mst_df)  # initialize mst & workload forecast models
		self.mst_tru = MstTru(mst_df)
		self.workload = self.read_workload(conf['workload_files'][conf['series']], 
										   conf['norm_factors'][conf['app']])
		self.timestep_sec = (self.workload[0:2].index[1] - self.workload[0:2].index[0]).total_seconds()
		self.backlog = 0
		self.conf = conf
		self.results = Results()
	

	def compute_backlog(self, m, workload, time):
		mst_tru = self.mst_tru.sample(m)
		delta = (workload - mst_tru) * time
		self.backlog = max(self.backlog + delta, 0)
		return (self.backlog, mst_tru)


	def start(self):
		# T = self.workload.size
		T = 50
		t = self.conf['t_start']
		t_report = T / self.conf['num_reports']
		m_curr = self.scalr.make_decision(self.workload[t], 1)
		startup_time = 0
		reconfig_time = 0
		cooldown_time = 0
		state = ScalrState.READY

		log.info('Starting simulation')

		while t < T:
			(backlog, mst_tru) = self.compute_backlog(m_curr, self.workload[t], self.timestep_sec)
			self.scalr.set_backlog(backlog)
			self.results.add(self.workload.index[t], m_curr, self.workload[t], mst_tru, backlog)

			log.debug('t={},\tstate={},\tm_curr={}'.format(t, state, m_curr))

			if state == ScalrState.READY:
				m_next = self.scalr.make_decision(self.workload[t], m_curr)
				if m_curr < m_next:
					state = ScalrState.STARTUP
					startup_time = self.conf['startup_time']
				elif m_curr > m_next:
					state = ScalrState.RECONFIG
					m_curr = 0
					reconfig_time = self.conf['reconfig_time']
				# if m_curr == m_next, stay READY state

			elif state == ScalrState.STARTUP:
				if 0 < startup_time:
					startup_time -= 1
				if startup_time == 0:
					state = ScalrState.RECONFIG					
					m_curr = 0
					reconfig_time = self.conf['reconfig_time']

			elif state == ScalrState.RECONFIG:
				if 0 < reconfig_time:
					reconfig_time -= 1
				if reconfig_time == 0:
					cooldown_time = self.conf['cooldown_time']
					state = ScalrState.COOLDOWN
					m_curr = m_next
					
			elif state == ScalrState.COOLDOWN:
				if 0 < cooldown_time:
					cooldown_time -= 1
				if cooldown_time == 0:
					state = ScalrState.READY

			else:
				log.error('Invalid state: {}'.format(str(state)))

			# if t % t_report == 0:
			# 	log.info('{}% ({}/{}) done'.format(round(100 * float(t)/T), t, T))
			t += 1

		log.info('Finished simulation')
		self.results.print_stats()
		
					
			
						
			
		
		
