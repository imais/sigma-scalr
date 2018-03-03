from enum import Enum

class InstanceState(Enum):
	NULL = -1
	PENDING = 0
	RUNNING = 16
	SHUTTING_DOWN = 32
	TERMINATED = 48
	STOPPING = 64
	STOPPED = 80
	
	def __str__(self):
		return self.name


class CloudManager(object):
	def __get_instances(self):

	def __init__(self, conf):
		self.conf = conf
		# retrieve initial instance info in 
		self.instances = self.__get_instances()


	def add_instances(self, num):
		# return newly added instances
		add_instances = 1

	def rm_instances(self, num):
		rm_instances = 1
		# return removed instances
		


		
