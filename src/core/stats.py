
class Stats(object):
	def __init__(self, startup_delay_sec, sched_interval_sec, vm_cost_per_hr):
		self.startup_delay_sec = startup_delay_sec
		self.sched_interval_sec = sched_interval_sec
		self.vm_cost_per_hr = vm_cost_per_hr
		self.violations_sec = 0.0
		self.m_mape = 0.0
		self.mst_up_mape = 0.0
		self.mst_op_mape = 0.0
		self.mst_mape = 0.0
		self.backlog = 0.0
		self.max_backlog = 0.0
		self.total_backlog = 0.0
		self.total_m = 0.0
		self.total_time_sec = 0.0
		self.num_stats = 0
		self.stats_enabled = True

	def reset(self, stats_enabled):
		self.stats_enabled = stats_enabled
		self.violations_sec = 0.0
		self.m_mape = 0.0
		self.mst_up_mape = 0.0
		self.mst_op_mape = 0.0
		self.mst_mape = 0.0
		self.backlog = 0.0
		self.max_backlog = 0.0
		self.total_backlog = 0.0
		self.total_m = 0.0
		self.total_time_sec = 0.0
		self.num_stats = 0

	def update(self, mst_tru, demand_tru, m, m_tru, time_affected):
		if self.stats_enabled:
			diff = (float(abs(mst_tru - demand_tru)) / mst_tru) * \
				   time_affected / self.sched_interval_sec
			if mst_tru < demand_tru:
				self.violations_sec += time_affected
				self.mst_up_mape += diff
				self.backlog += (demand_tru - mst_tru) * time_affected
			else:
				self.mst_op_mape += diff
				self.backlog = max(0, self.backlog - (mst_tru - demand_tru) * self.startup_delay_sec)

			self.mst_mape += diff
			if self.max_backlog < self.backlog :
				self.max_backlog = self.backlog
				self.total_backlog += self.backlog

			self.total_m += m * time_affected / self.sched_interval_sec
			self.total_time_sec += time_affected

			self.m_mape += float(abs(m_tru - m) / m_tru) * \
						   (time_affected / self.sched_interval_sec) 


	def increment_num_stats(self):
		if self.stats_enabled:
			self.num_stats += 1


	def dump_stats(self):
		print("sched_interval_sec:\t {}".format(self.sched_interval_sec))
		print("violations_sec:\t\t {}".format(self.violations_sec))
		print("m_mape:\t\t\t {}".format(self.m_mape))
		print("mst_up_mape:\t\t {}".format(self.mst_up_mape))
		print("mst_op_mape:\t\t {}".format(self.mst_op_mape))
		print("mst_mape:\t\t {}".format(self.mst_mape))
		print("backlog:\t\t {}".format(self.backlog))
		print("max_backlog:\t\t {}".format(self.max_backlog))
		print("total_backlog:\t\t {}".format(self.total_backlog))
		print("total_m:\t\t {}".format(self.total_m))
		print("total_time_sec:\t\t {}".format(self.total_time_sec))
		print("vm_cost_per_hr:\t\t {}".format(self.vm_cost_per_hr))
		print("num_stats:\t\t {}".format(self.num_stats))


	def print_stats(self):
		if self.stats_enabled:
			print("Stats: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f" % \
				  (100. * (1.0 - (self.violations_sec/self.total_time_sec)), \
				   100. * (self.mst_mape / self.num_stats), \
				   100. * (self.mst_up_mape / self.num_stats), \
				   100. * (self.mst_op_mape / self.num_stats), \
				   self.total_m, \
				   self.total_m / self.num_stats, \
				   self.vm_cost_per_hr * self.total_m / self.num_stats, \
				   100 * (self.m_mape / self.num_stats), \
				   # we need to divide total_backlog by 2*num_stats since backlog is added twice per loop
				   self.total_backlog, \
				   (self.total_backlog / (2 * self.num_stats)), \
				   self.max_backlog))		


