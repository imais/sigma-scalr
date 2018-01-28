import logging
from enum import Enum
from random import randint

log = logging.getLogger(__name__)


class ScalrState(Enum):
	NULL = 0
	READY = 1
	STARTUP = 2
	RECONFIG = 3
	COOLDOWN = 4
	strings = {NULL: 'NULL', READY: 'READY', STARTUP: 'STARTUP',
			   RECONFIG: 'RECONFIG', COOLDOWN: 'COOLDOWN'}

	def __str__(self):
		return self.name
		

class Scalr(object):
	def __init__(self, conf, mst_df):
		log.debug('Scalr initialized')
		self.n = 0


	def make_decision(self, workload, m_curr):
		m = m_curr
		if self.n == 10:
			m = max(m_curr + randint(-2, 2), 1)
		else:
			self.n += 1
		return m


	def set_backlog(self, backlog):
		self.backlog = backlog
			
		

		
	
