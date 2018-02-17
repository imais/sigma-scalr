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
			return self.effective_mst_dict[(m_curr,m_next,S,D,L)]

		# m vs. time:
		# m: | m_curr |  0  | m_next |
		# t: |<--S--->|<-D->| 
		# t: |<----------L---------->|
		total_mst = self.mst_model.predict(m_curr) * S
		total_mst += self.mst_model.predict(m_next) * (L - (S + D))
		effective_mst = total_mst / L
		self.effective_mst_dict[(m_curr,m_next,S,D,L)] = effective_mst
		return effective_mst


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
				mst_pred = self.__compute_effective_mst(m_curr, m, S, D, L)
			else:
				mst_pred = self.mst_model.predict(m)
			delta = mst_pred - workload_pred

			variance = 0.0
			if self.conf['mst_uncertainty_aware']:
				variance += self.mst_model.std**2
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


	def __estimate_backlog(self, backlog, workload, workload_std, m, time):
		alpha = 0.0
		beta = 0.0
		if self.conf['backlog_uncertainty_aware']:
			alpha = workload_std		# 1 sigma = 68%
			beta = self.mst_model.std	# 1 sigma = 68%
			
		delta = ((workload + alpha) - max(self.mst_model.predict(m) - beta, 0)) * time
		return max(backlog + delta, 0)


	def __estimate_m_backlog_aware(self, workload, m_curr, op):
		assert((op == ScalrOp.DOWN) or (op == ScalrOp.UP))

		forecast, ci, std = self.__forecast_workload(self.conf['lookahead_steps'])

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)

		backlog = self.backlog
		backlog_list = [backlog]
		S = self.conf['startup_steps'] if op == ScalrOp.UP else 0
		D = self.conf['reconfig_steps'] # 'D'owntime
		L = self.conf['lookahead_steps']
		
		# [0, S) VM startup 
		for t in range(0, S):
			# TODO: consider std when computing delta
			backlog = self.__estimate_backlog(backlog, forecast[t], std[t], m_curr, 
											  self.conf['timestep_sec'])
			backlog_list.append(backlog)

		# [S, S+D): reconfiguration
		for t in range(S, S+D):
			backlog = self.__estimate_backlog(backlog, forecast[t], std[t], 0, 
											  self.conf['timestep_sec'])
			backlog_list.append(backlog)

		# [S+D, L): after reconfiguration
		for m in range(1, self.mst_model.m_max+1):
			# TODO: search from m_curr, not from 1
			b = backlog
			m_found = False
			backlog_list_ = []
			for t in range(S+D, L):
				b = self.__estimate_backlog(b, forecast[t], std[t], 
											m, self.conf['timestep_sec'])
				backlog_list_.append(b)
				if b <= 0:
					m_found = True
					break
			if m_found:
				break
		
		if m_found:
			log.debug('\tBacklog expected be 0 in {} steps with m={}: backlog={}'.
					  format(t+1, m, backlog_list + backlog_list_))
		else:
			log.warn('\tBacklog not expected to  be 0 in {} steps even with max m={}: backlog={}'.
					 format(t+1, m, backlog_list + backlog_list_))
		
		return m if m_found else self.mst_model.m_max


	def __estimate_m_forecast_uncertainty_aware(self, workload, m_curr):
		forecast, ci, std = self.__forecast_workload(self.conf['lookahead_steps'])

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)
		
		# pick up the max workload in the next lookahead steps
		upper_lim = ci['upper y']
		max_index = upper_lim.loc[upper_lim == upper_lim.max()].index[0]
		workload = forecast[max_index]
		workload_std = std[max_index]

		return self.__estimate_m(workload, workload_std, m_curr, op)

	
	def make_decision(self, workload, m_curr):
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
				m = self.__estimate_m_backlog_aware(workload, m_curr, op)
			elif self.conf['forecast_uncertainty_aware']:
				m = self.__estimate_m_forecast_uncertainty_aware(workload, m_curr)
			else:
				m = self.__estimate_m(workload)

		return m, op


	def put_backlog(self, backlog):
		self.__put_backlog(backlog)
			
	def put_workload(self, workload):
		self.__put_workload(workload)
		

		
	
