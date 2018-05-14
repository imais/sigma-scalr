import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import scipy
from scipy import stats

cpu_util_file = "../data/cpu_util/raw_cpu_util.tsv"
app = 'grep'
df_all = pd.DataFrame.from_csv(cpu_util_file, sep='\t', header=0)
df = df_all.loc[df_all['app'] == app]

# dist_names = [ 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy'] 

# # Search for the distribution that gives the lowest average p-value
# avg_pvalues = {}
# for dist_name in dist_names:
# 	dist = getattr(scipy.stats, dist_name)
# 	print('Testing {}...'.format(dist_name))

# 	pvalues = []
# 	for index, row in df.iterrows():
# 		non_nan_filter = [not np.isnan(v) for v in row[2:]]
# 		cpu_utils = pd.to_numeric(row[2:][non_nan_filter], errors='coerce').values

# 		params = dist.fit(cpu_utils)
# 		_, pvalue = stats.kstest(cpu_utils, dist_name, args=[p for p in params])
# 		pvalues.append(pvalue)

# 	avg_pvalues[dist_name] = np.mean(pvalues)

# print avg_pvalues
# print max(avg_pvalues, key=avg_pvalues.get)

# {'genhalflogistic': 0.04611840647496336, 'wald': 0.04206109958999268, 'mielke': 0.09712161422630909, 'triang': 0.027204791966028683, 'pareto': 0.028990365768721212, 'kstwobign': 0.038709498498456424, 'rayleigh': 0.04200887578212369, 'arcsine': 0.012545428122927452, 'fisk': 0.08931877530826805, 'betaprime': 0.02803257873890974, 'frechet_l': 0.07973437106261655, 'genpareto': 0.008549245168992085, 'genexpon': 0.04164075165336967, 'rice': 0.02056603954275975, 'lomax': 0.008439193489491077, 'wrapcauchy': np.nan, 'frechet_r': 0.050105819674249805, 'foldnorm': 0.05600580594174703, 'tukeylambda': 0.03458109760093144, 'dgamma': 0.1648229590830205, 'dweibull': 0.1459103899632897, 'ksone': np.nan, 'gilbrat': 0.04402790104127668, 'loglaplace': 0.06394135852373117, 'gumbel_r': 0.05173765803723614, 'lognorm': 0.05140125567194332, 'semicircular': 0.01857090245404953, 'exponweib': 0.04849408397193067, 'halfcauchy': 0.0429617779849362, 'chi': 0.03219171825747562, 'invweibull': 0.05179556304308108, 'exponpow': 0.05564231842795182, 'genlogistic': 0.09628760458324043, 'cosine': 0.039828704756578195, 'truncexpon': 0.0025622898847353064, 'anglit': 0.02960011866477, 'weibull_max': 0.07973437106261655, 'maxwell': 0.04606915282297748, 'weibull_min': 0.050105819674249805, 'truncnorm': 0.0, 'expon': 0.02843741887851426, 'invgauss': 0.04693314924496999, 'norm': 0.059019596535647895, 'gumbel_l': 0.07847813304045215, 'recipinvgauss': np.nan, 'halfnorm': 0.026865184754178303, 'fatiguelife': 0.04796217437606754, 'halflogistic': 0.033572258157021945, 'chi2': 0.021040985587575944, 'gengamma': 0.03571804571682085, 'johnsonsu': 0.11204450185666402, 'uniform': 0.013840816779252633, 'foldcauchy': 0.038016762662697684, 'gausshyper': 0.029104032050220534, 'powernorm': 0.0587804216938637, 'beta': 0.05437179351010092, 't': 0.08204819535648407, 'genextreme': 0.08989648094971703, 'powerlaw': 0.022230689008256133, 'burr': 0.05955040277300075, 'alpha': 0.039820742182923634, 'johnsonsb': 0.05021362614226336, 'hypsecant': 0.10127326037897562, 'ncf': 0.04684633324314317, 'bradford': 0.004876332382884678, 'nakagami': 0.05080158682879633, 'erlang': 0.055170722006306405, 'gompertz': 0.04815375002698797, 'reciprocal': 0.0, 'f': 0.0451690134902761, 'cauchy': 0.075724430446335, 'vonmises': np.nan, 'loggamma': 5.711773824409526e-06, 'pearson3': np.nan, 'nct': 0.027737110945824952, 'logistic': 0.0915210423617042, 'invgamma': 0.051702440256542484, 'powerlognorm': 0.06300264579485242, 'ncx2': 0.008732221770831295, 'rdist': 0.03925689046981399, 'laplace': 0.07028321302745746, 'gamma': 0.03379433858243992}
# dgamma

i = 1
for index, row in df.iterrows():
	non_nan_filter = [not np.isnan(v) for v in row[2:]]
	cpu_utils = pd.to_numeric(row[2:][non_nan_filter], errors='coerce').values
	lnspc = np.linspace(0, 100, 10000)
	
	plt.subplot(3, 6, i)
	plt.title(str(row.vm) + ' VMs')	
	# h = plt.hist(cpu_utils, normed=True, alpha=0.5)
	plt.hist(cpu_utils, bins=range(100), normed=True, alpha=0.8, label='samples')

	m, s = stats.norm.fit(cpu_utils)
	pdf = stats.norm.pdf(lnspc, m, s)
	if np.isnan(pdf).any():
		print('{}\tNaN!!!'.format(row.vm))	
	plt.plot(lnspc, pdf, label='norm', color='blue')

	# a, b, c = stats.dgamma.fit(cpu_utils)
	# print('{}\t{}\t{}\t{}'.format(row.vm, a, b, c))
	# pdf = stats.dgamma.pdf(lnspc, a, b, c)
	# plt.plot(lnspc, pdf, label='dgamma')

	# a, loc, scale = stats.alpha.fit(cpu_utils)
	# pdf = stats.alpha.pdf(lnspc, a, loc=loc, scale=scale)
	# plt.plot(lnspc, pdf, label='alpha')

	# ag, bg, cg = stats.gamma.fit(cpu_utils)
	# pdf = stats.gamma.pdf(lnspc, ag, bg, cg)
	# plt.plot(lnspc, pdf, label='gamma')

	# ab, bb, cb, db = stats.beta.fit(cpu_utils)
	# pdf = stats.beta.pdf(lnspc, ab, bb, cb, db)
	# plt.plot(lnspc, pdf, label='beta')

	loc, c, scale = stats.dweibull.fit(cpu_utils)
	# print('{}\t{}\t{}\t{}'.format(row.vm, loc, c, scale))
	pdf = stats.dweibull.pdf(lnspc, loc, c, scale)
	if np.isnan(pdf).any():
		print('{}\tNaN!!!'.format(row.vm))
	plt.plot(lnspc, pdf, label='dweibull', color='green')

	i += 1

plt.tight_layout(pad=0.4, w_pad=0.25, h_pad=1.0)
plt.xlabel('CPU Utilization [%]')
plt.ylabel('Frequency')
plt.legend(bbox_to_anchor=(1.8, 1), loc=2, borderaxespad=0.)
plt.show()	
		



