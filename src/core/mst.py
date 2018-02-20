import logging
import numpy as np
import sys
import pandas as pd
from lsqnonneg import lsqnonneg
from scipy.stats import norm

log = logging.getLogger()


class MstModel(object):
	SUPPORTED_MODELS = ['model1', 'model2']

	def __init__(self, data_file, model_file, model_id, app):
		if not model_id in self.SUPPORTED_MODELS:
			sys.exit("Specified model {} not supported".format(model_id))
		self.id = model_id

		mst_df = self.__read_mst_data(data_file, app)
		training_subset = self.__read_mst_training_subset(model_file, model_id, app)
		filtered_df = mst_df.loc[mst_df['m'].isin(training_subset)]
		filtered_df = pd.melt(filtered_df.iloc[:, 1:], id_vars='m')

		self.m_max = max(mst_df['m'])
		self.m_train = np.array(filtered_df.loc[:, 'm'])
		self.mst_train = np.array(filtered_df.loc[:, 'value'])
		self.num_training = 0
		self.ready = False
		self.__train()

		log.info('model_id={}, app={}, weights={}, std={}'
				 .format(model_id, app, self.weights, self.std))

		
	def __read_mst_data(self, data_file, app):
         df = pd.DataFrame.from_csv(data_file, sep='\t', header=0)
         return df.loc[df['app'] == app]


	def __read_mst_training_subset(self, model_file, model_id, app):
		df = pd.DataFrame.from_csv(model_file, sep='\t', header=0)
		subset_df = df.loc[(df['model'] == model_id) & (df['app'] == app), 'training_subset']
		subset_csv = subset_df.iloc[0] # dataframe to string
		return np.array([float(w) for w in subset_csv.split(",")])


	def __train(self):
		if self.id == 'model1':
			Z = np.array([[1., 1./x, x, x**2] for x in self.m_train])
			y = np.array([1/y for y in self.mst_train])
		elif self.id == 'model2':
			Z = np.array([[1., x, -(x**2)] for x in self.m_train])			
			y = self.mst_train
		residual = .0
		try:
			(weights, resnorm, residual) = lsqnonneg(Z, y, itmax_factor=10)
		except Exception:
			# Ignore the iteration count exceeded exception
			pass
		else:
			self.weights = weights
			self.resnorm = resnorm
			self.residual = residual
			self.mst = self.__compute_mst(self.m_train)
			self.std = self.__compute_std(self.m_train, self.mst_train)
			self.num_training += 1



	def __compute_mst(self, m):
		mst_peak = 0
		mst = [0]  # mst(0) = 0
		for m_ in range(1, self.m_max+1):
			mst_peak = max(self.__compute(m_), mst_peak)
			mst.append(mst_peak)
		self.ready = True
		return mst


	def __compute(self, m):
		if self.id == 'model1':
			z = np.array([1., 1./m, m, m**2])
			mst = 1 / np.dot(z, self.weights)
		elif self.id == 'model2':
			z = np.array([1., m, -m**2])
			mst = np.dot(z, self.weights)
		else:
			mst = 0.0
		return mst


	def __compute_std(self, m, mst):
		# compute predicion variance: SSE/(n-k)
		mst_hat = [self.predict(m_) for m_ in m]
		sse = 0.0
		sse += ((mst - mst_hat)**2).sum()
		n = len(mst)
		k = len(self.weights)
		return np.sqrt(sse/(n - k))


	def update(self, m_train, mst_train):
		# Assume both m & mst are float numbers
		self.m_max = max(m_train, self.m_max)
		self.m_train = np.append(self.m_train, m_train)
		self.mst_train = np.append(self.mst_train, mst_train)
		self.train()


	def predict(self, m):
		return self.mst[m] if self.ready else 1
