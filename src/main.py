import argparse
import json
import logging
import logging.config
import os
import time
from ast import literal_eval as make_tuple
from sim.sim import Sim
from realtime.realtime import RealTime

log = logging.getLogger()


def setup_logger(path='./conf/log.json', level=logging.INFO):
	if os.path.exists(path):
		with open(path, 'rt') as f:
			conf = json.load(f)
		logging.config.dictConfig(conf)
	else:
		logging.basicConfig(level=level)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a',	'--app', default='grep', type=str)
	parser.add_argument('-s',	'--series', default='wcup', type=str)
	parser.add_argument('-m',	'--model', default='model1', type=str)
	parser.add_argument('-p',	'--arima_pdq', default='(1,0,0)', type=str)
	parser.add_argument('-r',	'--rho', default=0.95, type=float)
	parser.add_argument('-cf',	'--conf', default='./conf/conf.json', type=str)
	parser.add_argument('-sim', '--simulation', action='store_true')
	# proposed policies
	parser.add_argument('-mua',	'--mst_uncertainty_aware', action='store_true')
	parser.add_argument('-fua', '--forecast_uncertainty_aware', action='store_true')
	parser.add_argument('-ol',  '--online_learning', action='store_true')	
	parser.add_argument('-ba',	'--backlog_aware', action='store_true')
	parser.add_argument('-bap',	'--backlog_aware_proactive', action='store_true')
	# step-scaling policies
	parser.add_argument('-sav', '--step_scaling_avg', action='store_true')
	parser.add_argument('-smn', '--step_scaling_min', action='store_true')
	parser.add_argument('-smx', '--step_scaling_max', action='store_true')
	# ground truth & static scheduling policies
	parser.add_argument('-st',  '--static_scheduling', action='store_true')
	parser.add_argument('-gt',  '--ground_truth', action='store_true')

	args = parser.parse_args()

	return args


def init_conf(args):
	if os.path.exists(args.conf):
		with open(args.conf, 'rt') as f:
			# args have priority over settings in conf
			conf = dict(json.load(f).items() + vars(args).items())
			conf['arima_pdq'] = make_tuple(conf['arima_pdq'])
	else:
		conf = args

	results_dir = "./results/"
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	sched_opt = ('_mua' if conf['mst_uncertainty_aware'] else '') + \
				('_ol'  if conf['online_learning'] else '') + \
				('_fua' if conf['forecast_uncertainty_aware'] else '') + \
				('_ba'  if conf['backlog_aware'] else '') + \
				('_bap' if conf['backlog_aware_proactive'] else '') + \
				('_sav' if conf['step_scaling_avg'] else '') + \
				('_smn' if conf['step_scaling_min'] else '') + \
				('_smx' if conf['step_scaling_max'] else '') + \
				('_st'  if conf['static_scheduling'] else '') + \
				('_gt'  if conf['ground_truth'] else '')
	conf['sched_opt'] = sched_opt[1:] if 1 <= len(sched_opt) else 'none'
	results_file = results_dir + str(int(time.time())) + \
				   '_' + conf['app'] + \
				   '_rho=' + str(args.rho) + '_' + conf['sched_opt'] + \
				   ".tsv"
	conf['results_file'] = results_file

	# enforce flag dependencies
	if conf['backlog_aware_proactive']:
		conf['backlog_aware'] = True

	if conf['step_scaling_avg'] or conf['step_scaling_min'] or conf['step_scaling_max']:
		conf['step_scaling'] = True
	else:
		conf['step_scaling'] = False

	return conf


def check_conf(conf):
	pass


def main(conf):
	log.info('Configs: {}'.format(conf))
	if conf['simulation']:
		s = Sim(conf)
		s.start()
	else:
		r = RealTime(conf)
		r.start()

	
if __name__ == '__main__':
	setup_logger()
	args = parse_args()
	conf = init_conf(args)
	check_conf(conf)
	main(conf)
