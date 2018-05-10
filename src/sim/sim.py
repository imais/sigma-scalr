# from __future__ import absolute_import

import logging
import pandas as pd
import datetime
import numpy as np
from util.results import Results
from core.scalr import Scalr, ScalrState, ScalrOp
from mst_tru import MstTru
from cpu_util import CpuUtil


log = logging.getLogger()


class Sim(object):
	def read_workload(self, workload_file, norm_factor):
		workload = pd.Series.from_csv(workload_file, sep='\t', header=0)
		if norm_factor < 0:
			return workload
		else:
			scale = norm_factor / workload.max()
			return workload.multiply(scale)

	def __estimate_m_truth(self, workload):
		for m in range(1, self.m_max):
			if workload <= self.mst_tru.predict(m):
				break
		return m

		
	def __init__(self, conf):
		self.mst_tru = MstTru(conf['mst_data_file'], conf['app'])
		self.cpu_util = CpuUtil(conf['cpu_util_file'], conf['app'], self.mst_tru)		
		self.workload = self.read_workload(conf['workload_files'][conf['series']], 
										   conf['norm_factors'][conf['app']])
		self.timestep_sec = int((self.workload[0:2].index[1] - self.workload[0:2].index[0]).total_seconds())
		conf['timestep_sec'] = self.timestep_sec
		conf['scheduling_interval_sec'] = self.timestep_sec * conf['scheduling_interval_steps']
		self.scalr = Scalr(conf)
		self.m_max = self.scalr.mst_model.m_max
		self.m_truth = self.__estimate_m_truth(max(self.workload))
		self.conf = conf
		self.results = Results(conf['sched_opt'], conf['app'])


	def __step_scaling(self, backlog, workload, m_curr):
		cpu_samples = [self.cpu_util.sample(m_curr, workload, backlog) for i in range(0, m_curr)]
		cpu_util = 0.0
		if self.conf['step_scaling_min']:
			cpu_util = min(cpu_samples)
		elif self.conf['step_scaling_max']:
			cpu_util = max(cpu_samples)
		else:
			cpu_util = np.mean(cpu_samples)

		for idx, cpu_util_bound in enumerate(self.conf['step_scaling_conf']['cpu_util_bounds']):
			if cpu_util <= cpu_util_bound:
				break
		scaling_adjustment = float(self.conf['step_scaling_conf']['scaling_adjustments'][idx-1]) / 100.
		m_next = int(round((1.0 + scaling_adjustment) * m_curr))
		m_next = min(max(m_next, 1), self.m_max)

		if m_curr < m_next:
			op = ScalrOp.UP
		elif m_curr > m_next:
			op = ScalrOp.DOWN
		else:
			op = ScalrOp.NULL
		
		return m_next, op, cpu_util


	def __ground_truth(self, backlog, future_workload, m_curr):
		# objective: to find the minimum m that makes 0 backlog at the end of this scheduling cycle		
		# future_workload holds future actual input data rates until the end of this scheduling cycle
		timestep_sec = self.conf['timestep_sec']

		for m_next in range(1, self.m_max+1):
			if self.__compute_future_backlog(backlog, future_workload, m_curr, m_next) == 0:							break
			
		if m_curr < m_next:
			op = ScalrOp.UP
		elif m_curr > m_next:
			op = ScalrOp.DOWN
		else:
			op = ScalrOp.NULL
		
		return m_next, op


	def __compute_future_backlog(self, backlog, workload, m_curr, m_next):
		t = 0
		startup_sec = 0
		reconfig_sec = 0

		if m_curr == m_next:
			# No change
			state = ScalrState.WORK
			backlog_sec = self.conf['timestep_sec']
		elif m_curr > m_next:
			# Scale down: no VM startup time
			state = ScalrState.RECONFIG
			reconfig_sec = self.conf['reconfig_sec']
		else:
			# Scale up: VM startup time 
			state = ScalrState.STARTUP
			startup_sec = self.conf['startup_sec']

		while t < self.conf['scheduling_interval_steps']-1:
			timestep_sec = self.conf['timestep_sec']

			if state == ScalrState.STARTUP:
				if timestep_sec < startup_sec:
					startup_sec -= timestep_sec
					backlog_sec = timestep_sec
					timestep_sec = 0					
				else:
					state = ScalrState.RECONFIG
					timestep_sec -= startup_sec
					reconfig_sec = self.conf['reconfig_sec']
					backlog_sec = startup_sec
				backlog, _ = self.compute_backlog(backlog, workload[t], m_curr, backlog_sec)

			if state == ScalrState.RECONFIG and 0 < timestep_sec:
				if timestep_sec < reconfig_sec:
					reconfig_sec -= timestep_sec
					backlog_sec = timestep_sec					
					timestep_sec = 0
				else:
					state = ScalrState.WORK
					timestep_sec -= reconfig_sec
					backlog_sec = reconfig_sec
				backlog, _ = self.compute_backlog(backlog, workload[t], 0, backlog_sec)					
				
			if state == ScalrState.WORK and 0 < timestep_sec:
				backlog, _ = self.compute_backlog(backlog, workload[t], m_next, backlog_sec)

			t += 1
			
		return backlog


	def compute_backlog(self, backlog, workload, m, time):
		mst_tru = self.mst_tru.sample(m)
		delta = (workload - mst_tru) * time
		backlog = max(backlog + delta, 0)
		return (backlog, mst_tru)


	def log(self, timestamp, workload, timedelta_sec, state_pre, state_cur,
			m_curr, mst_tru, backlog_sec, backlog):
		log.debug(' {}s,\t{}->{}, m_curr={}, backlog_sec={}, mst={}, backlog={}'
				  .format(timedelta_sec, state_pre, state_cur, m_curr, backlog_sec, mst_tru, backlog))
		self.results.add(timestamp, m_curr, workload, mst_tru, backlog)


	def start(self):
		log.info('Starting simulated scheduling')

		T = self.workload.size - self.conf['scheduling_interval_steps']
		t = self.conf['t_sim_start']
		for i in range(0, t):
			self.scalr.put_workload(self.workload[i])
		t_report = T / self.conf['num_reports']
		m_curr = self.m_truth if self.conf['static_scheduling'] else 3 
		startup_sec = 0
		reconfig_sec = 0
		interval_sec = 0
		backlog = 0
		cpu_util = 0.0
		state = ScalrState.READY

		while t < T:
			log.debug('t={},\t{}, m_curr={}, workload={}, backlog={}'
					  .format(t, state, m_curr, self.workload[t], backlog))
			
			timestep_sec = self.timestep_sec
			timestamp = self.workload.index[t]
			self.scalr.put_workload(self.workload[t])

			# NOTE: state can change multiple times within one timestep
			# we make scaling decisions only when we are in READY state			
			if state == ScalrState.READY:
				if self.conf['ground_truth']:
					t_next = t + self.conf['scheduling_interval_steps'] - 1
					m_next, op = self.__ground_truth(backlog, self.workload[t:t_next], m_curr)
				elif self.conf['step_scaling']:
					m_next, op, cpu_util = self.__step_scaling(backlog, self.workload[t], m_curr)
				elif self.conf['static_scheduling']:
					m_next, op = self.m_truth, ScalrOp.NULL
				else:
					m_next, op = self.scalr.make_decision(backlog, self.workload[t], m_curr)
					
				log.debug('\t### Scaling decision: m_next={}, op={}'.format(m_next, op))
				if op != ScalrOp.NULL:
					state = ScalrState.STARTUP
					startup_sec = self.conf['startup_sec']
				else:
					state = ScalrState.WORK
				interval_sec = self.conf['scheduling_interval_sec']

			if state == ScalrState.STARTUP:
				state_pre = state
				if timestep_sec < startup_sec:
					startup_sec -= timestep_sec
					backlog_sec = timestep_sec
					timestep_sec = 0
				else:
					timestep_sec -= startup_sec
					backlog_sec = startup_sec					
					state = ScalrState.RECONFIG
					reconfig_sec = self.conf['reconfig_sec']
				interval_sec -= backlog_sec
				backlog, mst_tru = self.compute_backlog(backlog, self.workload[t], m_curr, backlog_sec)
				timedelta_sec = self.timestep_sec - timestep_sec
				# self.log(timestamp, self.workload[t], timedelta_sec, state_pre, state,
				# 		 m_curr, mst_tru, backlog_sec, backlog)
				timestamp += datetime.timedelta(0, timedelta_sec)				

			if state == ScalrState.RECONFIG and 0 < timestep_sec:
				state_pre = state
				m_curr = m_next				
				if timestep_sec < reconfig_sec:
					reconfig_sec -= timestep_sec
					backlog_sec = timestep_sec					
					timestep_sec = 0
				else:
					state = ScalrState.WORK
					timestep_sec -= reconfig_sec
					backlog_sec = reconfig_sec
				interval_sec -= backlog_sec					
				backlog, mst_tru = self.compute_backlog(backlog, self.workload[t], 0, backlog_sec)
				timedelta_sec = self.timestep_sec - timestep_sec
				# self.log(timestamp, self.workload[t], timedelta_sec, state_pre, state,
				# 		 m_curr, mst_tru, backlog_sec, backlog)
				timestamp += datetime.timedelta(0, timedelta_sec)

			if state == ScalrState.WORK and 0 < timestep_sec:
				state_pre = state
				interval_sec -= timestep_sec

				# even if there is not enough interval_sec left, we wait until next timestep
				backlog_sec = timestep_sec
				timestep_sec = 0				
				backlog, mst_tru = self.compute_backlog(backlog, self.workload[t], m_curr, backlog_sec)
				timedelta_sec = self.timestep_sec - timestep_sec

				if interval_sec <= 0:
					state = ScalrState.READY				
					if self.conf['online_learning'] and mst_tru < self.workload[t]:
						self.scalr.mst_model_update(m_curr, mst_tru)
				
			self.scalr.put_backlog(backlog)	# backlog at the end of time t
			self.results.add(self.workload.index[t], t,
							 state, m_curr, self.workload[t], mst_tru, backlog, cpu_util)
			if t % t_report == 0:
				log.info('{}% ({}/{}) done'.format(round(100 * float(t)/T), t, T))
			t += 1

		log.info('Finished simulation')
		self.results.print_stats()
		if 'results_file' in self.conf:
			log.info('Writing results to {}'.format(self.conf['results_file']))
			self.results.write_results(self.conf['results_file'])
		
					
			
						
			
		
		
