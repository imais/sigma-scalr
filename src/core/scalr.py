import logging
import Queue
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


	def __compute_effective_mst(self, m_curr, m_next, S, D, L):
		# S: Startup time, D: Downtime, L: Lookahead time
		if (m_curr,m_next,S,D,L) in self.effective_mst_dict:
			return self.effective_mst_dict[(m_curr,m_next,S,D,L)], self.effective_mst_dict['std']
		elif (m_next,D,L) in self.effective_mst_dict:
			return self.effective_mst_dict[(m_next,D,L)], self.effective_mst_dict['std']

		# m vs. time:
		# m: | m_curr |  0  | m_next |
		# t: |<--S--->|<-D->| 
		# t: |<----------L---------->|
		total_mst = 0 if S == 0 else self.mst_model.predict(m_curr) * S
		total_mst += self.mst_model.predict(m_next) * (L - (S + D))
		effective_mst = total_mst / L

		if 'std' not in self.effective_mst_dict:
			# https://stats.stackexchange.com/questions/168971/variance-of-an-average-of-random-variables
			var = ((self.mst_model.std**2) * (L - D)) / L**2
			self.effective_mst_dict['std'] = np.sqrt(var)

		if S == 0:
			# ignore m_curr & S
			self.effective_mst_dict[(m_next,D,L)] = effective_mst
		else:
			self.effective_mst_dict[(m_curr,m_next,S,D,L)] = effective_mst
		return effective_mst, self.effective_mst_dict['std']


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
		return forecast, ci, std
		

	def __estimate_m(self, workload_pred, workload_std=0.0, m_curr=0):
		for m in range(1, self.mst_model.m_max+1):
			if self.conf['forecast_effective_mst']:
				S = self.conf['startup_steps'] if m > m_curr else 0
				D = self.conf['reconfig_steps']
				L = self.conf['lookahead_steps']
				mst_pred, mst_std = self.__compute_effective_mst(m_curr, m, S, D, L)
			else:
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
		alpha = 0.0
		beta = 0.0
		if self.conf['backlog_uncertainty_aware']:
			alpha = workload_std		# 1 sigma = 68%
			beta = self.mst_model.std	# 1 sigma = 68%
			
		delta = ((workload_forecast + alpha) - max(self.mst_model.predict(m) - beta, 0)) * time
		return max(backlog + delta, 0)


	def __estimate_future_backlogs(self, backlog, workload_forecast, workload_std, m_curr, m_next):
		backlog_list = [backlog]
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

		cooldown = False
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
				cooldown = True

			if cooldown:
				backlog_list.append(backlog)
			t += 1
			
		return backlog_list

	
	def __estimate_m_backlog_aware(self, workload, backlog, m_curr):
		forecast, ci, std = self.__forecast_workload(self.conf['scheduling_interval_steps'])
		B = self.conf['target_backlog']

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)


		# first, search downward
		if 1 < m_curr:
			min_m, m_found = m_curr-1, False
			for m in range(m_curr-1, 0, -1):
				# backlog_list only contains backlogs after m is active
				backlog_list = self.__estimate_future_backlogs(backlog, forecast, std, m_curr, m)
				if max(backlog_list) <= B:
					min_m = m
					m_found = True
				else:
					break
			if m_found:
				return min_m

		# next, search upward (including m_curr)
		backlog_max_prev = -1.0
		backlog_list_dict = {}
		m_found = False
		for m in range(m_curr, self.mst_model.m_max+1):
			backlog_list = self.__estimate_future_backlogs(backlog, forecast, std, m_curr, m)
			backlog_list_dict[m] = backlog_list
			backlog_max = max(backlog_list)
			if backlog_max == backlog_max_prev:
				# no more improvment, exit with prev m
				m = m - 1
				m_found = True
				break
			elif backlog_max <= B:
				m_found = True
				break
			backlog_max_prev = backlog_max

		# if not m_found:
		# 	for m in range(m_curr+1, self.mst_model.m_max+1):
		# 		if backlog_list_dict[m][-1] == 0:
		# 			m_found = True
		# 			break
			
		return m


	def __estimate_m_backlog_aware_avg(self, workload, backlog, m_curr):
		forecast, ci, std = self.__forecast_workload(self.conf['scheduling_interval_steps'])
		B = self.conf['target_backlog']

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)


		# first, search downward
		if 1 < m_curr:
			min_m, m_found = m_curr-1, False
			for m in range(m_curr-1, 0, -1):
				backlog_list = self.__estimate_future_backlogs(backlog, forecast, std, m_curr, m)
				if np.mean(backlog_list) <= B:
					min_m = m
					m_found = True
				else:
					break
			if m_found:
				return min_m

		# next, search upward (including m_curr)
		backlog_avg_prev = -1.0
		backlog_list_dict = {}
		m_found = False
		for m in range(m_curr, self.mst_model.m_max+1):
			backlog_list = self.__estimate_future_backlogs(backlog, forecast, std, m_curr, m)
			backlog_list_dict[m] = backlog_list
			backlog_avg = np.mean(backlog_list)
			if backlog_avg == backlog_avg_prev:
				# no more improvment, exit with prev m
				m = m - 1
				m_found = True
				break
			elif backlog_avg <= B:
				m_found = True
				break
			backlog_avg_prev = backlog_avg

		# if not m_found:
		# 	for m in range(m_curr+1, self.mst_model.m_max+1):
		# 		if backlog_list_dict[m][-1] == 0:
		# 			m_found = True
		# 			break
			
		return m	


	def __estimate_m_forecast_uncertainty_aware(self, workload, m_curr):
		forecast, ci, std = self.__forecast_workload(self.conf['scheduling_interval_steps'])

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)
		
		# pick up the max workload in the next lookahead steps
		upper_lim = ci['upper y']
		max_index = upper_lim.loc[upper_lim == upper_lim.max()].index[0]
		workload = forecast[max_index]
		workload_std = std[max_index]

		return self.__estimate_m(workload, workload_std, m_curr)

	
	def make_decision(self, workload, backlog, m_curr):
		m = m_curr

		if self.conf['fixed_interval_scheduling']:
			op = ScalrOp.NULL
		else:
			# check scaling trigger conditions from backlog
			op = self.__check_scaling_trigger() 

		if op == ScalrOp.DOWN and m_curr == 1:
			log.debug('\tIgnore DOWN operation since m_curr=1')
		else:
			if self.conf['backlog_aware']:
				m = self.__estimate_m_backlog_aware(workload, backlog, m_curr)
			elif self.conf['forecast_uncertainty_aware']:
				m = self.__estimate_m_forecast_uncertainty_aware(workload, m_curr)
			else:
				m = self.__estimate_m(workload)

		return m, op


	def put_backlog(self, backlog):
		self.__put_backlog(backlog)

		
	def put_workload(self, workload):
		self.__put_workload(workload)


	def mst_model_update(self, m, workload):
		self.mst_model.update(m, workload)
