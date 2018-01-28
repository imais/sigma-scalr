import sys
import numpy as np
from lsqnonneg import lsqnonneg
from scipy.stats import norm

class MstModel(object):
	supported_models = ['model1', 'model2']

	def __init__(self, id, m_train, mst_train, m_max):
		if not id in self.supported_models:
			sys.exit("Specified model {} not supported".format(id))
		self.id = id
		self.m_max = m_max
		self.m_train = m_train
		self.mst_train = mst_train
		self.num_training = 0
		self.train()

		
	def train(self):
		if self.id == 'model1':
			Z = np.array([[1., 1./x, x, x**2] for x in self.m_train])
			y = np.array([1/y for y in self.mst_train])
		elif self.id == 'model2':
			Z = np.array([[1., x, -(x**2)] for x in self.m_train])			
			y = self.mst_train
		try:
			(weights, resnorm, residual) = lsqnonneg(Z, y, itmax_factor=10)
		except Exception:
			# Ignore the iteration count exceeded exception
			pass
		else:
			self.weights = weights
			self.mst = self.compute_mst(self.m_train)
			self.std = self.compute_std(self.m_train, self.mst_train)
			self.num_training += 1


	def update(self, m_train, mst_train):
		# Assume both m & mst are float numbers
		self.m_max = max(m_train, self.m_max)
		self.m_train = np.append(self.m_train, m_train)
		self.mst_train = np.append(self.mst_train, mst_train)
		self.train()

	def compute_mst(self, m):
		mst_peak = 0
		mst = [0]  # mst(0) = 0
		for m_ in range(1, self.m_max+1):
			mst_peak = max(self.compute(m_), mst_peak)
			mst.append(mst_peak)
		return mst


	def compute_std(self, m, mst):
		# compute predicion variance: SSE/(n-k)
		mst_hat = [self.predict(m_) for m_ in m]
		sse = 0.0
		sse += ((mst - mst_hat)**2).sum()
		n = len(mst)
		k = len(self.weights)
		return np.sqrt(sse/(n - k))


	def compute(self, m):
		if self.id == 'model1':
			z = np.array([1., 1./m, m, m**2])
			mst = 1 / np.dot(z, self.weights)
		elif self.id == 'model2':
			z = np.array([1., m, -m**2])
			mst = np.dot(z, self.weights)
		else:
			mst = 0.0
		return mst


	def predict(self, m):
		return self.mst[m]


class MstTru(object):
	# ground truth computation - take average and linear interpolation? or actually run all the configs?

	def __init__(self, mst_df, random_sampling=True):
		# compute mean over all samples
		mst_mean = mst_df.iloc[:, 2:].mean(axis=1) 
		mst_std = mst_df.iloc[:, 2:].std(axis=1)
		self.m_max = max(mst_df['m'])

		# linearly interpolate mean and std, both arrays are 0-based
		self.mst_mean = np.interp(range(0, self.m_max + 1), mst_df['m'], mst_mean)
		self.mst_std = np.interp(range(0, self.m_max + 1), mst_df['m'], mst_std)

		self.random_sampling = random_sampling


	def sample(self, m):
		if m < 0 or self.m_max < m:
			return 0
		elif self.random_sampling:
			while True:
				rnd = norm.rvs(loc=self.mst_mean[m - 1], scale=self.mst_std[m - 1], size=1)
				if (0 < rnd):
					break;
			return rnd
		else:
			return self.mst_mean[m - 1] 

