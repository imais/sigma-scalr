import statsmodels.api as sm

class ARIMA(object):
	stationarity = False
	invertibility = False

	def __init__(self, order, seasonal_order=(0,0,0,0)):
		self.order = order
		self.seasonal_order = seasonal_order


	def forecast(self, data, steps_ahead, order=None, seasonal_order=None):
		if order is None:
			order = self.order
		if seasonal_order is None:
			seasonal_order = self.seasonal_order

		try:
			if self.seasonal_order == (0,0,0,0):
				model = sm.tsa.statespace.SARIMAX(data,
												  order=order,
												  enforce_stationarity=ARIMA.stationarity,
												  enforce_invertibility=ARIMA.invertibility)
			else:
				model = sm.tsa.statespace.SARIMAX(data,
												  order=order,
												  seasonal_order=seasonal_order,
												  enforce_stationarity=ARIMA.stationarity,
												  enforce_invertibility=ARIMA.invertibility)
			self.results = model.fit(disp=False)
		except:
			pass # do nothing

		# print("model.nobs={}, freq={}".format(model.nobs, self.results.model.data.freq))
		pred = self.results.get_prediction(start=model.nobs, end=model.nobs + steps_ahead - 1)
		pred_val = pred.predicted_mean
		pred_ci = pred.conf_int(alpha=0.05)
		upper_lim = pred_ci.iloc[:, 1]
		# 1.96 is z* for 95% confidence interval
		std = (upper_lim - pred_val) / 1.96
		
		return pred_val, pred_ci, std
