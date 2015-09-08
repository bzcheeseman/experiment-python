import sys

sys.path.append("..")

from basics import get_data, error_code

import os

cwd = os.getcwd()

class fitter(object):

	def __init__(self, path_to_data, dataset, model, save_name = None, 
		datatype = "csv", h5labels = [None, None, None], 
		path_to_errtext = cwd+"/..", debug = False):

		self.data = get_data(path_to_data = path_to_data, dataset = dataset, datatype = datatype, 
			h5xlabel = h5labels[0], h5ylabel = h5labels[1], h5zlabel = h5labels[2],
			path_to_errtext = path_to_errtext, path_to_library = path_to_errtext)

		self.model = model
		if save_name is not None:
			self.save_name = save_name

	def choose_domain(self, xdata, ydata, domain, upper = None):
		import numpy as np

		if upper != None:
			locs = np.where(ydata >= upper)[0]
			for index in locs:
				ydata[index] = upper

		lower_bound = np.where(xdata == domain[0])[0]
		upper_bound = np.where(xdata == domain[1])[0]

		return xdata[lower_bound:upper_bound], ydata[lower_bound:upper_bound]

	def find_peaks(self, domain, std_dev = 11):
		from scipy.ndimage.filters import gaussian_filter
		from scipy.signal import argrelmin
		import numpy as np

		xdata = self.data[:,0]
		ydata = self.data[:,1]

		xpeaks, ypeaks = self.choose_domain(xdata, ydata, domain)

		ygauss = gaussian_filter(ypeaks, std_dev)

		x_peak_coord = xpeaks[argrelmin(ygauss)[0]]
		y_peak_coord = ypeaks[argrelmin(ygauss)[0]]

		return x_peak_coord, y_peak_coord

	def single_lorentzian(self, parms, plot = True, err = 1e-10, maxruns = int(1e8), 
		xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$"):

		from scipy.optimize import curve_fit
		import numpy as np
		import matplotlib.pyplot as plt

		xdata = self.data[:,0]
		ydata = self.data[:,1]

		yerr = np.multiply(.001, ydata)

		def lorentzian(x, c, A, G, w0):
			return c + (A/np.pi) * (.5*G)/((x-w0)**2+(.5*G)**2)

		popt, pcov = curve_fit(lorentzian, xdata, ydata, parms, 
			check_finite = True, xtol = err, maxfev = maxruns,
			sigma = yerr)

		chi_square = np.sum(lorentzian(xdata, *popt)-ydata)/(yerr**2)

		yFit = lorentzian(xdata, *popt)
		yuFit = lorentzian(xdata, *parms)

		if plot == True:
			plt.figure(figsize = (7,7))
			plt.plot(xdata, ydata, 'o', ms = .3, label = r"$Raw \, Data$")
			plt.plot(xdata, yFit, linewidth = 4, alpha = .6, label = r"$Fitted \, Curve$")
			plt.plot(xdata, yuFit, alpha = .8, label = r"$Guesses$")
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			plt.legend(loc = 0)
			if self.save_name is not None:
				plt.savefig(self.save_name+".png")
			plt.show()
		else:
			pass

		error_code(code = 0, path_to_errtext = "..")
		return popt, chi_square, yFit, yuFit

	def multi_lorentzian(self, parms, nres, domain, std_dev = 11, plot = True, err = 1e-10, maxruns = int(1e8), 
		xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$"):

		from scipy.optimize import curve_fit
		import numpy as np
		import matplotlib.pyplot as plt

		xdata = self.data[:,0]
		ydata = self.data[:,1]

		x_peak_coord, y_peak_coord = self.find_peaks(domain, std_dev)

		print len(x_peak_coord), len(y_peak_coord)

		parms[1+nres:2*nres+1] = y_peak_coord.flatten()
		parms[2*nres+1:] = x_peak_coord.flatten()

		parms = parms.flatten()

		yerr = .01*np.ones_like(ydata)

		def multi_lorentz(x, *p):
			y = np.zeros_like(x) + p[0]
			for i in range(1, len(p)-2*nres):
				gamma = p[i]
				amp = p[i+nres]
				center = p[i+2*nres]
				y -= np.absolute(amp/(1+1.j*(x-center)/(gamma/2)))
			return y.flatten()

		popt, pcov = curve_fit(multi_lorentz, xdata, ydata, parms, 
			check_finite = True, xtol = err, maxfev = maxruns)

		chi_square = np.sum((-multi_lorentz(xdata, *popt)+ydata)**2/(yerr**2))/(len(xdata) - len(popt) - 1)

		yFit = multi_lorentz(xdata, *popt)
		yuFit = multi_lorentz(xdata, *parms)

		if plot == True:
			plt.figure(figsize = (7,7))
			plt.plot(xdata, ydata, 'o', ms = .3, label = r"$Raw \, Data$")
			plt.plot(xdata, yFit, linewidth = 4, alpha = .6, label = r"$Fitted \, Curve$")
			plt.plot(xdata, yuFit, alpha = .8, label = r"$Guesses$")
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			plt.legend(loc = 0) ##include a save_name
			plt.show()
		else:
			pass

		error_code(0)
		return popt, chi_square, yFit, yuFit

	def calc_Q(self, parms, nres):
		import numpy as np

		qs = np.absolute(np.divide(parms[1+2*nres:], parms[1:nres+1]))
		print qs

		return qs


if __name__ == "__main__":
	print "make sure all the inputs are in the right order..."














