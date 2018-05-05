import logging
import Queue
import math
import numpy as np
from enum import Enum
from random import randint
from core.mst import MstModel
from core.arima import ARIMA
from scipy.stats import norm

log = logging.getLogger()


class ScalrState(Enum):
	NULL = 0
	READY = 1
	STARTUP = 2
	RECONFIG = 3
	COOLDOWN = 4

	def __str__(self):
		return self.name


class ScalrOp(Enum):
	NULL = 0
	UP = 1
	DOWN = 2

	def __str__(self):
		return self.name
		

class Scalr(object):
	def __init__(self, conf):
		self.mst_model = MstModel(conf['mst_data_file'], conf['mst_model_file'], 
								  conf['model'], conf['app'])
		self.arima = ARIMA(conf['arima_pdq'])
		self.workload_window = Queue.Queue()
		self.backlog_window = Queue.Queue()
		self.conf = conf
		self.effective_mst_dict = {}


	def __put_workload(self, workload):
		self.workload_window.put(workload)
		if self.conf['workload_window_size'] < self.workload_window.qsize():
			self.workload_window.get()


	def __put_backlog(self, backlog):
		self.backlog = backlog
		self.backlog_window.put(backlog)
		if self.conf['backlog_window_size'] < self.backlog_window.qsize():
			# pop one item to keep the queue length backlog_window_size
			self.backlog_window.get()


	def __check_scaling_trigger(self):
		op = ScalrOp.NULL

		if self.backlog_window.qsize() == 0:
			log.debug('\tcheck_scaling: no backlog yet, op={}'.format(op))
			return op

		mean_backlog = np.mean(list(self.backlog_window.queue))

		if self.conf['backlog_scaleup_threshold'] < mean_backlog:
			op = ScalrOp.UP
		elif mean_backlog < self.conf['backlog_scaledown_threshold']:
			op = ScalrOp.DOWN
		log.debug('\tcheck_scaling: mean_backlog={}, op={}'.format(mean_backlog, op))

		return op


	def __forecast_workload(self, L):
		# make L-step lookahead workload forecast
		if self.workload_window.qsize() < self.conf['workload_window_size']:
			# if we don't have enough window data, use AR(1)
			forecast, ci, std = self.arima.forecast(self.workload_window.queue, L, (1,0,0))
		else:
			forecast, ci, std = self.arima.forecast(self.workload_window.queue, L)
		# put the current workload in front of the forecast
		current_workload = np.array([self.workload_window.queue[-1]])
		return np.append(current_workload, forecast), ci, std
		

	def __estimate_m(self, workload_pred, workload_std=0.0):
		for m in range(1, self.mst_model.m_max+1):
			mst_pred, mst_std = self.mst_model.predict(m), self.mst_model.std
			delta = mst_pred - workload_pred

			variance = 0.0
			if self.conf['mst_uncertainty_aware']:
				variance += mst_std**2
			if self.conf['forecast_uncertainty_aware']:
				variance += workload_std**2

			if 0 < variance:
				std = np.sqrt(variance)
				prob = norm.sf(x=0, loc=delta, scale=std)   # survival function: 1-cdf
				if self.conf['rho'] <= prob:
					break
			elif 0 <= delta:
				# if uncertainty is not considered, 0 <= delta
				break;
                        
		return m


	def __estimate_backlog(self, backlog, workload_forecast, workload_std, m, time):
		delta = (workload_forecast - self.mst_model.predict(m)) * time
		return max(backlog + delta, 0)


	def __estimate_future_backlogs(self, backlog, workload_forecast, workload_std, m_curr, m_next):
		backlog_list = []
		t = 0
		startup_sec = 0
		reconfig_sec = 0

		if m_curr == m_next:
			# No change
			state = ScalrState.COOLDOWN
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
				backlog = self.__estimate_backlog(backlog, workload_forecast[t], workload_std[t],
												  m_curr, backlog_sec)

			if state == ScalrState.RECONFIG and 0 < timestep_sec:
				if timestep_sec < reconfig_sec:
					reconfig_sec -= timestep_sec
					backlog_sec = timestep_sec					
					timestep_sec = 0
				else:
					state = ScalrState.COOLDOWN
					timestep_sec -= reconfig_sec
					backlog_sec = reconfig_sec
				backlog = self.__estimate_backlog(backlog, workload_forecast[t], workload_std[t],
												  0, backlog_sec)
				
			if state == ScalrState.COOLDOWN and 0 < timestep_sec:
				backlog = self.__estimate_backlog(backlog, workload_forecast[t], workload_std[t],
												  m_next, timestep_sec)

			backlog_list.append(backlog)
			t += 1
			
		return backlog_list

	
	def __estimate_m_backlog_aware(self, workload, backlog, m_curr):
		forecast, ci, std = self.__forecast_workload(self.conf['scheduling_interval_steps']-1)
		B = 0  # target backlog at the end of this scheduling cycle

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)

		# first, search downward
		if 1 < m_curr:
			min_m, m_found = m_curr-1, False
			for m in range(m_curr-1, 0, -1):
				# backlog_list only contains backlogs after m is active
				backlog_list = self.__estimate_future_backlogs(backlog, forecast, std, m_curr, m)
				if backlog_list[-1] == B:
					min_m = m
					m_found = True
				else:
					break
			if m_found:
				return min_m

		# next, search upward (including m_curr)
		m_found = False
		for m in range(m_curr, self.mst_model.m_max+1):
			backlog_list = self.__estimate_future_backlogs(backlog, forecast, std, m_curr, m)
			if backlog_list[-1] == B:
				break
			
		return m


	def __estimate_reconfig_backlogs(self, backlog, workload_forecast, workload_std, m_curr, op):
		backlog_list = []
		t = 0
		T = 0
		startup_sec = self.conf['startup_sec']
		reconfig_sec = self.conf['reconfig_sec']
		timestep_sec = self.conf['timestep_sec']


		if op == ScalrOp.UP:
			state = ScalrState.STARTUP
			T = ((startup_sec + reconfig_sec) / timestep_sec) + 1
		elif op == ScalrOp.DOWN:
			state = ScalrState.RECONFIG
			T = (reconfig_sec / timestep_sec) + 1
		else:
			# op == ScalrOp.NULL
			return [backlog]

		while t < T:
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
				backlog = self.__estimate_backlog(backlog, workload_forecast[t], workload_std[t],
												  m_curr, backlog_sec)

			if state == ScalrState.RECONFIG and 0 < timestep_sec:
				if timestep_sec < reconfig_sec:
					reconfig_sec -= timestep_sec
					backlog_sec = timestep_sec					
					timestep_sec = 0
				else:
					timestep_sec -= reconfig_sec
					backlog_sec = reconfig_sec
				backlog = self.__estimate_backlog(backlog, workload_forecast[t], workload_std[t],
												  0, backlog_sec)
				
			backlog_list.append(backlog)
			t += 1
			
		return backlog_list
	
	
	def __estimate_m_backlog_amortize(self, workload, backlog, m_curr):
		forecast, ci, std = self.__forecast_workload(self.conf['scheduling_interval_steps']-1)

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)
		
		# pick up the max workload in the next lookahead steps
		upper_lim = ci['upper y']
		max_index = upper_lim.loc[upper_lim == upper_lim.max()].index[0]
		workload = forecast[max_index]
		workload_std = std[max_index]

		# first, search downward
		backlog_list = self.__estimate_reconfig_backlogs(backlog, forecast,
														 std, m_curr, ScalrOp.DOWN)
		amortize_sec = self.conf['scheduling_interval_steps'] * self.conf['timestep_sec'] - \
					   self.conf['reconfig_sec']
		workload_adjusted = workload + (backlog_list[-1] / amortize_sec)
		m = self.__estimate_m(workload_adjusted, workload_std)
		if m < m_curr:
			return m
		
		# next, check if m_curr can cover the highest workload
		amortize_sec = self.conf['scheduling_interval_steps'] * self.conf['timestep_sec']
		workload_adjusted = workload + (backlog / amortize_sec)
		m = self.__estimate_m(workload_adjusted, workload_std)
		if m == m_curr:
			return m
	
		# third, search upward
		backlog_list = self.__estimate_reconfig_backlogs(backlog, forecast,
														 std, m_curr, ScalrOp.UP)
		amortize_sec = self.conf['scheduling_interval_steps'] * self.conf['timestep_sec'] - \
					   (self.conf['startup_sec'] + self.conf['reconfig_sec'])
		workload_adjusted = workload + (backlog_list[-1] / amortize_sec)
		# worst case, we will return m_max		
		m = self.__estimate_m(workload_adjusted, workload_std)

		return m
	

	def __estimate_m_forecast_uncertainty_aware(self, workload, m_curr):
		forecast, ci, std = self.__forecast_workload(self.conf['scheduling_interval_steps']-1)

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)
		
		# pick up the max workload in the next lookahead steps
		upper_lim = ci['upper y']
		max_index = upper_lim.loc[upper_lim == upper_lim.max()].index[0]
		workload = forecast[max_index]
		workload_std = std[max_index]

		return self.__estimate_m(workload, workload_std)

	
	def make_decision(self, workload, backlog, m_curr):
		if self.conf['backlog_aware']:
			backlog = 0 if self.conf['backlog_aware_proactive'] else backlog			
			m = self.__estimate_m_backlog_aware(workload, backlog, m_curr)
		elif self.conf['backlog_amortize']:
			backlog = 0 if self.conf['backlog_amortize_proactive'] else backlog
			m = self.__estimate_m_backlog_amortize(workload, backlog, m_curr)
		elif self.conf['forecast_uncertainty_aware']:
			m = self.__estimate_m_forecast_uncertainty_aware(workload, m_curr)
		else:
			m = self.__estimate_m(workload)

		return m


	def put_backlog(self, backlog):
		self.__put_backlog(backlog)

		
	def put_workload(self, workload):
		self.__put_workload(workload)


	def mst_model_update(self, m, workload):
		self.mst_model.update(m, workload)
