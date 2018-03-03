import logging
import pandas as pd
from threading import Thread, Lock 
from twisted.internet import reactor

from util.results import Results
from core.scalr import Scalr, ScalrState, ScalrOp
from metrics import MetricsServerClient, MetricsServerClientFactory


log = logging.getLogger()

# Global variables
val_store = {}
store_lock = Lock()

class MessageHandler(object):
	CMD_METRICS = 'metrics'
	CMD_RESOURCE_UPDATED = 'resource_updated'
	CMD_STREAMAPP_UPDATED = 'stream_app_updated'

	def set_val(self, var, val):
		# print('Acquiring lock')
		store_lock.acquire()
		try:
			val_store[var] = val
		finally:
			# print('Releasing lock')
			store_lock.release()


	def get_val(self, vars):
		vals = {}
		for var in vars:
			vals[var] = val_store[var] if var in val_store else None
		return vals


	def handle_msg(self, context, data):
		cmd = data['cmd']
		args = data['args']

		if cmd == CMD_METRICS:
			# set_val(args)
			# schedule next request
		# elif cmd == CMD_RESOURCE_UPDATED:
		# elif cmd == CMD_STREAMAPP_UPDATED:
		else:
			log.warn('Undefined command: {}'.format(cmd)



class RealTime(object):
	def __init__(self, conf):
		self.conf = conf
		self.msg_handler = MessageHandler()
		
	def start():
		log.info('Starting sigma-scalr')

		reactor.connectTCP(self.conf['metrics_server_host'], 
						   self.conf['metrics_server_port'], 
						   MetricsServerClientFactory(self.msg_handler))
		reactor.run()
		
		




