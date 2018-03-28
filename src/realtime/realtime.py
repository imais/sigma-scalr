import json
import logging
import pandas as pd
import time
import socket

from util.results import Results
from core.scalr import Scalr, ScalrState, ScalrOp
from cloud import CloudManager
from spe import SpeManager

BUFFER_SIZE = 1024

log = logging.getLogger()


class RealTime(object):
	def __init__(self, conf):
		self.conf = conf
		self.timestep_sec = self.conf['realtime']['timestep_sec']
		self.metrics = {}
		self.scalr = Scalr(conf)
		self.cloud_manager = CloudManager(conf)
		master_nodes = self.cloud_manager.get_instances(self.conf['realtime']['master_filter'])
		self.spe_manager = SpeManager(conf, self.cloud_manager, master_nodes[0])

		# connect to metrics server
		self.sock_metrics_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		host = self.conf['realtime']['metrics_server_host']
		port = self.conf['realtime']['metrics_server_port']
		self.sock_metrics_server.connect((host, port))
		log.info('Connected to metrics server {}:{}'.format(host, port))


	def __get_metrics(self):
		args = {'args': ['bytesin']}
		req = 'get ' + json.dumps(args)
		self.sock_metrics_server.send(req)
		resp = self.sock_metrics_server.recv(BUFFER_SIZE)
		resp = resp[-1] if resp.endswith('\n') else resp		

		if resp.startswith('ok'):
			self.metrics = json.loads(resp[3:])

			
	def __scale(self, m_next):
		# at least we keep one instance
		return self.cloud_manager.request_instances(max(m_next, 1))


		
	def start(self):
		log.info('Starting realtime scheduling')
		m_curr, op = 1, ScalrOp.NULL
		cooldown_steps = 0
		state = ScalrState.READY
		next_scheduling_time = 0
		
		while True:
			loop_start_sec = time.time()
			self.__get_metrics()
			workload = self.metrics['bytesin']	# could use 'bytesin_1min'
			self.scalr.put_workload(workload)
			log.info('workload: {}'.format(workload))

			# state machine
			if state == ScalrState.READY:
				# we make scaling decisions only when we are in READY state
				m_next, op = self.scalr.make_decision(workload, m_curr)
				log.debug('decision: m_next={}, op={}'.format(m_next, op))
				if m_curr < m_next:
					self.__scale(m_curr, m_next)
					state = ScalrState.STARTUP
					log.debug('### SCALING UP ###: {} -> {}'.format(m_curr, m_next))
				elif m_curr > m_next:
					self.__scale(m_curr, m_next)
					m_curr = m_next
					self.__reconfig_app()
					state = ScalrState.RECONFIG
					log.debug('### SCALING DOWN ###: {} -> {}'.format(m_curr, m_next))

				# if (not fixed_interval_scheduling) and m_curr == m_next, stay READY

				if self.conf['fixed_interval_scheduling'] or m_curr != m_next:
					if self.conf['fixed_interval_scheduling'] and m_curr == m_next:
						state = ScalrState.COOLDOWN
						log.debug('### COOLING DOWN ###')
					next_scheduling_time = time.time() + self.conf['scheduling_interval_sec']

			elif state == ScalrState.STARTUP:
				# since we do not want to block this event loop, poll instance states
				if self.cloud_manager.new_instances_started():
					m_curr = m_next
					self.spe_manager.reconfig()
					state = ScalrState.RECONFIG

			elif state == ScalrState.RECONFIG:
				if self.spe_manager.reconfig_done():
					state = ScalrState.COOLDOWN
					
			elif state == ScalrState.COOLDOWN:
				if time.time() > next_scheduling_time:
					state = ScalrState.READY

			else:
				log.error('Invalid state: {}'.format(str(state)))

			sleep_sec = self.timestep_sec - (time.time() - loop_start_sec)
			time.sleep(sleep_sec)


		# log.info('Finished simulation')
		# self.results.print_stats()
		# if 'results_file' in self.conf:
		# 	log.info('Writing results to {}'.format(self.conf['results_file']))
		# 	self.results.write_results(self.conf['results_file'])
			
			



		
		




