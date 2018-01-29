import argparse
import json
import logging
import logging.config
import os
from ast import literal_eval as make_tuple
from sim.sim import Sim

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
	parser.add_argument('-n',	'--num_tests', default=1, type=int)
	parser.add_argument('-r',	'--rho', default=0.95, type=float)
	parser.add_argument('-cf',	'--conf', default='./conf/conf.json', type=str)
	parser.add_argument('-sim', '--simulation', action='store_true')
	# scheduling options
	parser.add_argument('-mua',	'--mst_uncertainty_aware', action='store_true')
	parser.add_argument('-fua', '--forecast_uncertainty_aware', action='store_true')
	parser.add_argument('-ba', '--backlog_aware', action='store_true')
	parser.add_argument('-ol',  '--online_learning', action='store_true')
	args = parser.parse_args()
	args.arima_pdq = make_tuple(args.arima_pdq)

	return args


def init_conf(args):
	if os.path.exists(args.conf):
		with open(args.conf, 'rt') as f:
			# args have priority over settings in conf
			conf = dict(json.load(f).items() + vars(args).items())
	else:
		conf = args

	return conf


def main(conf):
	log.info('Configs: {}'.format(conf))
	if conf['simulation']:
		s = Sim(conf)
		s.start()
	else:
		log.info('Real-time mode to be implemented')

	
if __name__ == '__main__':
	setup_logger()
	args = parse_args()
	conf = init_conf(args)
	main(conf)
