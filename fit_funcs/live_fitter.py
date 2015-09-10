import sys

sys.path.append("..")

from basics import get_data, error_code

import os

cwd, _ = os.path.split(os.path.realpath(__file__))

class live_fitter(object):
	##for interfacing with the expt class in the DSL

	def __init__(self, expt):
		self.expt = expt

	def choose_domain(self, xdata, ydata, domain, upper = None):
		import numpy as np

		if upper != None:
			locs = np.where(ydata >= upper)[0]
			for index in locs:
				ydata[index] = upper

		lower_bound = np.where(xdata == domain[0])[0]
		upper_bound = np.where(xdata == domain[1])[0]

		return xdata[lower_bound:upper_bound], ydata[lower_bound:upper_bound]

	def find_peaks(self, xdata, ydata, domain, std_dev = 11):
		from scipy.ndimage.filters import gaussian_filter
		from scipy.signal import argrelmin
		import numpy as np

		xpeaks, ypeaks = self.choose_domain(xdata, ydata, domain)

		ygauss = gaussian_filter(ypeaks, std_dev)

		x_peak_coord = xpeaks[argrelmin(ygauss)[0]]
		y_peak_coord = ypeaks[argrelmin(ygauss)[0]]

		return x_peak_coord, y_peak_coord

	def single_lorentzian(self, xdata, ydata, parms, domain, upper = None, plot = True, err = 1e-10, maxruns = int(1e8), 
		xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$"):

		from scipy.optimize import curve_fit
		import numpy as np
		import matplotlib.pyplot as plt

		xpeaks, ypeaks = self.choose_domain(xdata, ydata, domain, upper = upper)

		yerr = .001*np.ones_like(ypeaks)

		def lorentzian(x, c, A, G, w0):
			return c + (A/np.pi) * (.5*G)/((x-w0)**2+(.5*G)**2)

		popt, pcov = curve_fit(lorentzian, xpeaks, ypeaks, parms, 
			check_finite = True, xtol = err, maxfev = maxruns,
			sigma = yerr)

		chi_square = np.sum((lorentzian(xpeaks, *popt)-ypeaks)**2/(yerr**2))/(len(xpeaks) - len(popt) - 1)

		yFit = lorentzian(xpeaks, *popt)
		yuFit = lorentzian(xpeaks, *parms)

		if plot == True:
			plt.figure(figsize = (7,7))
			plt.plot(xdata, ydata, 'o', ms = .3, label = r"$Raw \, Data$")
			plt.plot(xpeaks, yFit, linewidth = 4, alpha = .6, label = r"$Fitted \, Curve$")
			plt.plot(xpeaks, yuFit, alpha = .8, label = r"$Guesses$")
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			plt.legend(loc = 0)
			if self.save_name is not None:
				plt.savefig(self.save_name+".png")
			plt.show()
		else:
			pass

		os.chdir(cwd+'/..')
		error_code(code = 0)

		return popt, chi_square, yFit, yuFit

	def multi_lorentzian(self, xdata, ydata, parms, nres, domain, upper = .98, std_dev = 11, plot = True, err = 1e-10, maxruns = int(1e8), 
		xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$", outside_use = False):

		from scipy.optimize import curve_fit
		import numpy as np
		import matplotlib.pyplot as plt

		##Finding the peaks##

		xpeaks, ypeaks = self.choose_domain(xdata, ydata, domain, upper = upper)

		x_peak_coord, y_peak_coord = self.find_peaks(xdata, ydata, domain, std_dev)

		x_peak_coord = x_peak_coord.flatten()
		y_peak_coord = y_peak_coord.flatten()

		##Now we have the peaks, so we'll put them into the initial guesses##

		for i in range(0, len(y_peak_coord)):
			parms[1+nres+i] = parms[0]-y_peak_coord[i]
			parms[2*nres+1+i] = x_peak_coord[i]

		parms = parms.flatten()

		##And set the error...##UNDER CONSTRUCTION

		yerr = .001*np.ones_like(ypeaks)

		##And run the fit##

		def multi_lorentz(x, *p):
			y = np.zeros_like(x) + p[0]
			for i in range(1, len(p)-2*nres):
				gamma = p[i]
				amp = p[i+nres]
				center = p[i+2*nres]
				y -= np.absolute(amp/(1+1.j*(x-center)/(gamma/2)))
			return y.flatten()

		popt, pcov = curve_fit(multi_lorentz, xpeaks, ypeaks, parms, 
			check_finite = True, xtol = err, maxfev = maxruns)

		chi_square = np.sum((-multi_lorentz(xpeaks, *popt)+ypeaks)**2/(yerr)**2)/(len(xpeaks) - len(popt) - 1)

		yFit = multi_lorentz(xpeaks, *popt)
		yuFit = multi_lorentz(xpeaks, *parms)

		if plot == True:
			plt.figure(figsize = (7,7))
			plt.plot(xdata, ydata, 'o', ms = .3, label = r"$Raw \, Data$")
			plt.plot(xpeaks, yFit, linewidth = 4, alpha = .6, label = r"$Fitted \, Curve$")
			plt.plot(xpeaks, yuFit, alpha = .8, label = r"$Guesses$")
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			plt.legend(loc = 0) ##include a save_name?
			plt.show()
		else:
			pass

		os.chdir(cwd+'/..')

		error_code(code = 0)

		if outside_use != False:
			return popt, chi_square, yFit, yuFit, xpeaks
		else:
			return popt, chi_square, yFit, yuFit

	def calc_Q(self, parms, nres):
		import numpy as np

		qs = np.absolute(np.divide(parms[1+2*nres:], parms[1:nres+1]))
		print "The Q(s) is(are): ", qs

		return qs

	def weighted_avg(self, parms, nres):
		import numpy as np

		y_peaks = parms[1+nres:2*nres+1]
		x_peaks = parms[2*nres+1:]

		y_pks = np.sum(np.absolute(y_peaks))

		y_peaks = np.divide(y_peaks, y_pks)

		print "The weighted average frequency is:", np.average(x_peaks, weights = np.absolute(y_peaks))/1e9, "GHz"
		return np.average(x_peaks, weights = np.absolute(y_peaks))










