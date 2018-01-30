import logging
import Queue
import numpy as np
from enum import Enum
from random import randint
from core.mst import MstModel
from core.arima import ARIMA

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
		

	def __estimate_m(self, workload_pred, workload_std=0.0):
		for m in range(1, self.mst_model.m_max+1):
			mst_pred = self.mst_model.predict(m)
			delta = mst_pred - workload_pred

			variance = 0.0
			if self.conf['mst_uncertainty_aware']:
				variance += mst_model.std**2
			if self.conf['forecast_uncertainty_aware']:
				variance += workload_std**2

			if 0 < variance:
				std = np.sqrt(variance)
				prob = norm.sf(x=0, loc=delta, scale=std)   # survival function: 1-cdf
				# print("m={}, mst_mu={}, demand_mu={}, mu={}, std={}, prob={}, RHO={}".format(m, mst_mu, demand_mu, mu, std, prob, RHO))
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


	def __estimate_m_forecast_uncertainty_aware(self, workload):
		forecast, ci, std = self.__forecast_workload(self.conf['lookahead_steps'])

		if forecast is np.nan:
			log.warn('No forecast is returned')
			return self.__estimate_m(workload)
		
		# pick up the max workload in the next lookahead steps
		upper_lim = ci.iloc[:, 1]
		max_index = upper_lim.loc[upper_lim == upper_lim.max()].index[0]
		workload = forecast[max_index]
		workload_std = std[max_index]

		return self.__estimate_m(workload, workload_std)

	
	def make_decision(self, workload, m_curr):
		m = m_curr

		# check scaling trigger conditions from backlog
		op = self.__check_scaling_trigger()

		if op == ScalrOp.DOWN and m_curr == 1:
			log.debug('\tIgnore DOWN operation since m_curr=1')
		elif op == ScalrOp.UP or op == ScalrOp.DOWN:
			if self.conf['backlog_aware']:
				m = self.__estimate_m_backlog_aware(workload, m_curr, op)
			elif self.conf['forecast_uncertainty_aware']:
				m = self.__estimate_m_forecast_uncertainty_aware(workload)
			else:
				m = self.__estimate_m(workload)

		return m, op


	def put_backlog(self, backlog):
		self.__put_backlog(backlog)
			
	def put_workload(self, workload):
		self.__put_workload(workload)
		

		
	
