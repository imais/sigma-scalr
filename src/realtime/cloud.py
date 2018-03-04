import copy
import json
import os
from enum import Enum
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver


class CloudManager(object):

	def __update_running_instances(self):
		# list of 'libcloud.compute.base.Node' objects sorted in name
		self.all_instances = sorted(self.driver.list_nodes(ex_filters=self.base_filters)
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

		# we use stop/start existing instances for scaling (for now)
		self.base_filters = conf['realtime']['base_filters']
		self.__update_running_instances()
		

	def add_instances(self, num):
		base_index = len(self.running_instances)
		self.adding_instances = self.all_instances[base_index : base_index + num]
		result = True

		for i in self.adding_instances:
			if not self.driver.ex_start_node(i):
				log.error('Starting instance {} returned error'.format(i.id))
				result = False
				break

		return result
				

	def remove_instances(self, num):
		base_index = len(self.running_instances)
		self.removing_instances = self.all_instances[max(base_index - num, 0) : base_index]
		result = True

		for i in self.removing_instances:
			if not self.driver.ex_stop_node(i):
				log.error('Stopping instance {} returned error'.format(i.id))
				result = False
				break

		return result		


	def new_instances_all_running(self):
		instances_ids = [i.id for i in self.adding_instances]
		ex_filters = copy.copy(self.base_filters)
		ex_filters['instance-id'] = instances_ids

		adding_instances = sorted(self.driver.list_nodes(ex_filters=ex_filters), 
							   key=lambda node: node.name)

		running = True
		for i in adding_instances:
			if i.state != 'running':
				log.info('Instance {} is still in {} state'.format(i.id, i.state))
				running = False
				break
		
		if running:
			# TODO: we can avoid retrieving information for all nodes
			self.__update_running_instances()

		return running
			
		
			
		

		


		
