import statsmodels.api as sm

class TimeSeriesModel(object):
	stationarity = False
	invertibility = False

	def __init__(self, p, d, q, P, D, Q, s):
		if p != 0 or d != 0 or q != 0:
			self.order = (p, d, q)
		else:
			self.order = None
	
		if s != 0 and (P != 0 or D != 0 or Q != 0):
			self.seasonal_order = (P, D, Q, s)
		else:
			self.seasonal_order = None

	def __init__(self, order, seasonal_order):
		self.order = order
		self.seasonal_order = seasonal_order

	def forecast(self, data, order=None, seasonal_order=None):
		if order is None:
			order = self.order
		if seasonal_order is None:
			seasonal_order = self.seasonal_order

		try:
			if self.seasonal_order == (0,0,0,0):
				model = sm.tsa.statespace.SARIMAX(data,
												  order=order,
												  enforce_stationarity=TimeSeriesModel.stationarity,
												  enforce_invertibility=TimeSeriesModel.invertibility)
			else:
				model = sm.tsa.statespace.SARIMAX(data,
												  order=order,
												  seasonal_order=seasonal_order,
												  enforce_stationarity=TimeSeriesModel.stationarity,
												  enforce_invertibility=TimeSeriesModel.invertibility)
			self.results = model.fit(disp=False)
		# except Exception as e: 
		# 	print(e)
		except:
			pass # do nothing

		# one-step ahead forecast
		# print("model.nobs={}, freq={}".format(model.nobs, self.results.model.data.freq))
		pred = self.results.get_prediction(start=model.nobs, end=model.nobs)
		pred_val = pred.predicted_mean[0]
		pred_ci = pred.conf_int(alpha=0.05)
		upper_lim = pred_ci.iloc[:, 1][0]
		# 1.96 is z* for 95% confidence interval
		std = (upper_lim - pred_val) / 1.96
		
		return pred_val, std

	# def forecast(self, data):
	# 	return forecast(self, data, self.order, self.seasonal_order)

	# def forecast(self, data, order):
	# 	return forecast(self, data, order, self.seasonal_order)


