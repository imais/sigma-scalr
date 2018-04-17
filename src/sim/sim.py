# from __future__ import absolute_import

import logging
import pandas as pd
import datetime
from util.results import Results
from core.scalr import Scalr, ScalrState, ScalrOp
from mst_tru import MstTru


log = logging.getLogger()


class Sim(object):
	def read_workload(self, workload_file, norm_factor):
		workload = pd.Series.from_csv(workload_file, sep='\t', header=0)
		if norm_factor < 0:
			return workload
		else:
			scale = norm_factor / workload.max()
			return workload.multiply(scale)		


	def __init__(self, conf):
		self.mst_tru = MstTru(conf['mst_data_file'], conf['app'])
		self.workload = self.read_workload(conf['workload_files'][conf['series']], 
										   conf['norm_factors'][conf['app']])
		self.timestep_sec = int((self.workload[0:2].index[1] - self.workload[0:2].index[0]).total_seconds())
		conf['timestep_sec'] = self.timestep_sec # scalr needs this to estimate future backlog
		self.scalr = Scalr(conf)
		self.backlog = 0
		self.conf = conf
		self.results = Results()
	

	def compute_backlog(self, workload, m, time):
		mst_tru = self.mst_tru.sample(m)
		delta = (workload - mst_tru) * time
		self.backlog = max(self.backlog + delta, 0)
		return (self.backlog, mst_tru)


	def log(self, timestamp, workload, timedelta_sec, state_pre,
			state_cur, m_curr, mst_tru, backlog_sec, backlog):
		log.debug(' {}s,\t{}->{}, m_curr={}, backlog_sec={}, mst={}, backlog={}'
				  .format(timedelta_sec, state_pre, state_cur, m_curr, backlog_sec, mst_tru, backlog))
		self.results.add(timestamp + datetime.timedelta(0, timedelta_sec), m_curr,
						 workload, mst_tru, backlog)


	def start(self):
		log.info('Starting simulated scheduling')

		T = self.workload.size
		t = self.conf['t_sim_start']
		for i in range(0, t):
			self.scalr.put_workload(self.workload[i])
		t_report = T / self.conf['num_reports']
		m_curr, op = self.scalr.make_decision(self.workload[t], 1)
		startup_sec = 0
		reconfig_sec = 0
		cooldown_sec = 0
		backlog = 0
		state = ScalrState.READY

		while t < T:
			log.debug('t={},\t{}, m_curr={}, workload={}, backlog={}'
					  .format(t, state, m_curr, self.workload[t], backlog))
			
			timestep_sec = self.timestep_sec

			# NOTE: state can change multiple times within one timestep
			# we make scaling decisions only when we are in READY state			
			if state == ScalrState.READY:
				m_next, op = self.scalr.make_decision(self.workload[t], m_curr)
				log.debug('\tScaling decision: m_next={}, op={}'.format(m_next, op))
				if m_curr < m_next:
					state = ScalrState.STARTUP
					startup_sec = self.conf['startup_sec']
					log.debug('\t## SCALING UP: {} -> {}'.format(m_curr, m_next))
				elif m_curr > m_next:
					state = ScalrState.RECONFIG
					reconfig_sec = self.conf['reconfig_sec']
					log.debug('\t## SCALING DOWN: {} -> {}'.format(m_curr, m_next))
				elif self.conf['fixed_interval_scheduling']:
					state = ScalrState.COOLDOWN
				cooldown_sec = self.conf['cooldown_sec']					

			if state == ScalrState.STARTUP:
				state_pre = state
				if timestep_sec  < startup_sec:
					startup_sec -= timestep_sec
					backlog_sec = timestep_sec
					timestep_sec = 0					
				else:
					state = ScalrState.RECONFIG
					timestep_sec -= startup_sec
					reconfig_sec = self.conf['reconfig_sec']
					backlog_sec = startup_sec
				cooldown_sec -= backlog_sec
				backlog, mst_tru = self.compute_backlog(self.workload[t], m_curr, backlog_sec)
				timedelta_sec = self.timestep_sec - timestep_sec
				self.log(self.workload.index[t], self.workload[t], timedelta_sec,
						 state_pre, state, m_curr, mst_tru, backlog_sec, backlog)

			if state == ScalrState.RECONFIG and 0 <= timestep_sec:
				state_pre = state
				m_curr = m_next				
				if timestep_sec < reconfig_sec:
					reconfig_sec -= timestep_sec
					backlog_sec = timestep_sec					
					timestep_sec = 0
				else:
					state = ScalrState.COOLDOWN
					timestep_sec -= reconfig_sec
					backlog_sec = reconfig_sec
				cooldown_sec -= backlog_sec					
				backlog, mst_tru = self.compute_backlog(self.workload[t], 0, backlog_sec)
				timedelta_sec = self.timestep_sec - timestep_sec
				self.log(self.workload.index[t], self.workload[t], timedelta_sec,
						 state_pre, state, m_curr, mst_tru, backlog_sec, backlog)

			if state == ScalrState.COOLDOWN and 0 <= timestep_sec:
				state_pre = state
				cooldown_sec -= timestep_sec
				if cooldown_sec <= 0:
					state = ScalrState.READY
				# even if there is not enough cooldown_sec left, we wait until next timestep
				backlog_sec = timestep_sec
				timestep_sec = 0				
				backlog, mst_tru = self.compute_backlog(self.workload[t], m_curr, backlog_sec)
				timedelta_sec = self.timestep_sec - timestep_sec				
				self.log(self.workload.index[t], self.workload[t], timedelta_sec,
						 state_pre, state, m_curr, mst_tru, backlog_sec, backlog)				
				
			self.scalr.put_workload(self.workload[t])
			self.scalr.put_backlog(backlog)
			# self.results.add(self.workload.index[t], m_curr, self.workload[t], mst_tru, backlog)
			if t % t_report == 0:
				log.info('{}% ({}/{}) done'.format(round(100 * float(t)/T), t, T))
			t += 1

		log.info('Finished simulation')
		self.results.print_stats()
		if 'results_file' in self.conf:
			log.info('Writing results to {}'.format(self.conf['results_file']))
			self.results.write_results(self.conf['results_file'])
		
					
			
						
			
		
		
