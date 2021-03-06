# from __future__ import absolute_import

import logging
import pandas as pd
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
		self.timestep_sec = (self.workload[0:2].index[1] - self.workload[0:2].index[0]).total_seconds()
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


	def start(self):
		log.info('Starting simulation')

		T = self.workload.size
		t = self.conf['t_sim_start']
		t_report = T / self.conf['num_reports']
		m_curr, op = self.scalr.make_decision(self.workload[t], 1)
		startup_steps = 0
		reconfig_steps = 0
		cooldown_steps = 0
		state = ScalrState.READY

		while t < T:
			(backlog, mst_tru) = self.compute_backlog(self.workload[t], m_curr, self.timestep_sec)
			self.scalr.put_workload(self.workload[t])
			self.scalr.put_backlog(backlog)
			self.results.add(self.workload.index[t], m_curr, self.workload[t], mst_tru, backlog)

			log.debug('t={},\tstate={}, m_curr={}, workload={}, mst={}, backlog={}'.format(t, state, m_curr, self.workload[t], mst_tru, backlog))

			if state == ScalrState.READY:
				# we make scaling decisions only when we are in READY state
				m_next, op = self.scalr.make_decision(self.workload[t], m_curr)
				if op == ScalrOp.UP and m_curr < m_next:
					state = ScalrState.STARTUP
					startup_steps = self.conf['startup_steps']
					log.debug('\t### SCALING UP ###: {} -> {}'.format(m_curr, m_next))
				elif op == ScalrOp.DOWN and m_curr > m_next:
					state = ScalrState.RECONFIG
					reconfig_steps = self.conf['reconfig_steps']
					log.debug('\t### SCALING DOWN ###: {} -> {}'.format(m_curr, m_next))
					m_curr = 0
				# if m_curr == m_next, stay in READY state

			elif state == ScalrState.STARTUP:
				if 0 < startup_steps:
					startup_steps -= 1
				if startup_steps == 0:
					state = ScalrState.RECONFIG					
					m_curr = 0
					reconfig_steps = self.conf['reconfig_steps']

			elif state == ScalrState.RECONFIG:
				if 0 < reconfig_steps:
					reconfig_steps -= 1
				if reconfig_steps == 0:
					cooldown_steps = self.conf['cooldown_steps']
					state = ScalrState.COOLDOWN
					m_curr = m_next
					
			elif state == ScalrState.COOLDOWN:
				if 0 < cooldown_steps:
					cooldown_steps -= 1
				if cooldown_steps == 0:
					state = ScalrState.READY

			else:
				log.error('Invalid state: {}'.format(str(state)))

			if t % t_report == 0:
				log.info('{}% ({}/{}) done'.format(round(100 * float(t)/T), t, T))
			t += 1

		log.info('Finished simulation')
		self.results.print_stats()
		if 'results_file' in self.conf:
			log.info('Writing results to {}'.format(self.conf['results_file']))
			self.results.write_results(self.conf['results_file'])
		
					
			
						
			
		
		
