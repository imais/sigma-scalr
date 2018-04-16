import copy
import json
import os
from enum import Enum
from libcloud.compute.deployment import ScriptDeployment
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver

class RequestState(Enum):
	READY = 1
	IN_TRANSITION = 2

	def __str__(self):
		return self.name

	
class Request(Enum):
	NULL = 0
	START = 1
	STOP = 2
	FAILED = 100

	def __str__(self):
		return self.name	


class CloudManager(object):
	# all_instances		: instances returned from the driver using ex_filter
	# running_instances	: running instances (subset of all_instances)
	

	def __update_running_instances(self):
		# list of 'libcloud.compute.base.Node' objects sorted in name
		self.all_instances = sorted(self.driver.list_nodes(ex_filters=self.worker_filters),
									key=lambda node: node.name)
		# assuming at least one instance is running 
		self.running_instances = [inst for inst in self.all_instances if inst.state == 'running']
		

	def __init__(self, conf):
		self.conf = conf

		# only support Amazon EC2 (for now)
		assert(conf['realtime']['cloud_provider'] == "EC2")
		AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
		AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
		AWS_REGION = os.environ['AWS_REGION']
		EC2Driver = get_driver(Provider.EC2)
		self.driver = EC2Driver(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
		self.ssh_username = os.environ['EC2_USER']
		self.ssh_key_file = os.environ['EC2_SSH_KEY_FILE']

		# we use stop/start existing instances for scaling (for now)
		self.worker_filters = conf['realtime']['worker_filters']
		self.__update_running_instances()

		self.state = RequestState.READY
		self.num_requested_instances = 0
		self.last_request = Request.NULL
		

	def __add_instances(self, num):
		log.info('Starting {} instances'.format(num))
		
		base_index = len(self.running_instances)
		self.new_instances = self.all_instances[base_index : base_index + num]
		result = True

		for i in self.new_instances:
			if not self.driver.ex_start_node(i):
				log.error('Starting instance {} returned error'.format(i.id))
				result = False
				break

		return result
				

	def __stop_instances(self, num):
		log.info('Stopping {} instances'.format(num))
		
		base_index = len(self.running_instances)
		self.removing_instances = self.all_instances[max(base_index - num, 0) : base_index]
		result = True

		for i in self.removing_instances:
			if not self.driver.ex_stop_node(i):
				log.error('Stopping instance {} returned error'.format(i.id))
				result = False
				break

		return result


	# start or stop instances
	def request_instances(self, num_instances):
		# first, check if we need to make a new request
		if self.state == RequestState.READY and \
		   len(self.running_instances) == num_instances:
			log.info('Request in {} state, already {} instances running'
					 .format(self.state, num_instances))
			return true
		elif self.state == RequestState.IN_TRANSITION and \
		   self.num_requested_instances == num_instances:
			log.info('Request in {} state, already {} instances requested'
					 .format(self.state, num_instances))
			return true
		elif self.state == RequestState.IN_TRANSITION and \
			 self.num_requested_instances != num_instances:
			log.warn('Request in {} state, already requested {}, new request {})'
					 .format(self.state, self.num_requested_instances, num_instances))
			return false

		# now, we are in READY state and making a new request
		num_current_instances = len(self.running_instances)
		if num_instances > num_current_instances:
			request = Request.START			
			result = self.__start_instances(num_instances - num_current_instances)
		else:
			request = Request.STOP		
			result = self.__stop_instances(num_current_instances - num_instances)

		if result == True:
			self.num_requested_instances = num_instances
			self.last_request = request
		else:
			self.num_requested_instances = 0
			self.last_request = Request.FAILED			

		return result


	def new_instances_started(self):
		instances_ids = [i.id for i in self.new_instances]
		ex_filters = copy.copy(self.base_filters)
		ex_filters['instance-id'] = instances_ids

		new_instances = sorted(self.driver.list_nodes(ex_filters=ex_filters), 
									key=lambda node: node.name)

		running = True
		for i in new_instances:
			if i.state != 'running':
				log.info('Instance {} is still in {} state'.format(i.id, i.state))
				running = False
				break
		
		if running:
			# TODO: we can avoid retrieving information for all nodes
			self.__update_running_instances()

		return running


	def get_new_instances(self):
		return self.new_instances


	def get_instances(self, filter):
		try:
			self.driver
		except NameError:
			log.error('Driver not defined yet')
			return None

		return self.driver.list_nodes(ex_filters=filter)
		
		
	def run_cmd(self, instance, cmd):
		script = ScriptDeployment(cmd)
		return driver._connect_and_run_deployment_script(node=instance, task=script,
														 ssh_hostname=node.public_ips[0],
														 ssh_port=22,
														 ssh_username=self.ssh_username,
														 ssh_password='None',
														 ssh_key_file=self.ssh_key_file,
														 ssh_timeout=10, timeout=600, max_tries=3)
		

		


		
