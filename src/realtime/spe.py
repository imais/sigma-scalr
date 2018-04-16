import copy
import json
import os
import time
from enum import Enum
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver

# SPE: Stream Processing Engine


class SpeManager(object):
	def __init__(self, conf, cloud_manager, master_node):
		self.conf = conf
		self.slave_bootstrap_cmd = self.conf['realtime']['slave_bootstrap_cmd']
		self.master_bootstrap_cmd = self.conf['realtime']['master_bootstrap_cmd']
		self.cloud_manager = cloud_manager
		self.master_node = master_node
		self.cloud_manager.run_cmd(self.master_node, self.master_bootstrap_cmd)


	def reconfig(self):
		# first, execute the bootstrap cmd for the newly created instances
		new_instances = self.cloud_manager.get_new_instances()
		for i in new_instances:
			self.cloud_manager.run_cmd(i, self.slave_bootstrap_cmd)

		# ideally, check if all the new nodes have executed the bootstrap cmd
		# for now, we just wait for a few seconds
		time.sleep(3)

		# finally, resubmit the application


	# def reconfig_done(self):
		
