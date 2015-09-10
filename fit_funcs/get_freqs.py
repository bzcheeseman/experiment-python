import numpy as np
from fitter import *
from live_fitter import *
import matplotlib.pyplot as plt

class get_freqs(fitter):

	def __init__(self, name, path_to_data, dataset, guesses, nres, domain, upper = None): 
		## for a more comprehensive instance of fitter use the actual file or edit it here
		name = fitter(path_to_data = path_to_data, dataset = dataset)
		self.name = name
		
		data = name.data
		self.xdata = data[:,0]
		self.ydata = data[:,1]

		self.nres = nres
		self.upper = upper
		popt, chi, yFit, yuFit, xpeaks = name.multi_lorentzian(parms = guesses, nres = nres, domain = domain, upper = upper, plot = False, outside_use = True)
		self.popt = popt
		self.chi = chi
		self.yFit = yFit
		self.yuFit = yuFit
		self.xpeaks = xpeaks

		self.xAvg = name.weighted_avg(popt, nres)

	def get_freqs(self, xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$", disp_avg = True):
		parms = self.popt

		eps = np.sqrt(np.absolute(parms[self.nres+1:len(parms)-self.nres]))
		freqs = parms[1+2*self.nres:len(parms)]

		psi1 = np.divide(eps, np.sqrt(np.dot(eps, eps)))

		psicurr = psi1
		psiprev = np.zeros_like(psicurr)
		tprev = 0.

		E = []
		t = []

		def compute_next(psip, psipp, tp, w):
			deltnext = 0
			for i in range(0, len(w)):
				deltnext += float(w[i]*np.absolute(psip[i])**2)
			
			tnext = 0
			for j in range(0, len(w)):
				tnext += (((w[j] - deltnext) * psip[j]) - (tp * psipp[j]))**2

			tnext = tnext**.5
			
			psinext = np.zeros_like(psip)
			for k in range(0, len(w)):
				psinext[k] = (tnext**-1) * (psip[k]*(w[k]-deltnext)-tp*psipp[k])
			
			return deltnext, tnext, psinext

		i = 0
		while i < self.nres:
			dnext, tnxt, psinext = compute_next(psicurr, psiprev, tprev, freqs)
			psiprev = psicurr
			psicurr = psinext
			tprev = tnxt
			E.append(dnext)
			t.append(tnxt)
			i+=1

		xE = E
		yEmin = self.upper - max(self.popt[1+self.nres:2*self.nres+1])
		yEmax = self.upper+.01

		self.name.calc_Q(self.popt, self.nres)

		plt.figure(figsize = (7,7))
		plt.vlines(self.xAvg, yEmin, yEmax, alpha = .45, linewidth = 3, color = 'b', label = r"$Average$")
		plt.vlines(xE, yEmin, yEmax, linestyle = 'dotted', label = r"$Base \, Energies$")
		plt.plot(self.xdata, self.ydata, 'o', ms = .3, label = r"$Raw \, Data$")
		plt.plot(self.xpeaks, self.yFit, linewidth = 4, alpha = .6, label = r"$Fitted \, Curve$")
		plt.plot(self.xpeaks, self.yuFit, alpha = .8, label = r"$Guesses$")
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.legend(loc = 0)
		plt.show()

		print "E in GHz: ", np.divide(E, 1e9)
		print "t in MHz: ", np.divide(t, 1e6)
		return np.divide(E, 1e9), np.divide(t, 1e6), self.popt, self.chi

class get_freqs_live(live_fitter):

	def __init__(self, name, expt, nres):
		name = live_fitter(expt)
		self.name = name
		self.nres = nres

	def get_freqs_live(self, xdata, ydata, parms, domain, upper = .98, liveplot = True, 
		xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$"):

		parms, chi, yFit, yuFit, xpeaks = self.name.multi_lorentzian(xdata, ydata, parms, self.nres, domain, plot = False, outside_use = True)
		xAvg = self.name.weighted_avg(parms, self.nres)

		eps = np.sqrt(np.absolute(parms[self.nres+1:len(parms)-self.nres]))
		freqs = parms[1+2*self.nres:len(parms)]

		psi1 = np.divide(eps, np.sqrt(np.dot(eps, eps)))

		psicurr = psi1
		psiprev = np.zeros_like(psicurr)
		tprev = 0.

		E = []
		t = []

		def compute_next(psip, psipp, tp, w):
			deltnext = 0
			for i in range(0, len(w)):
				deltnext += float(w[i]*np.absolute(psip[i])**2)
			
			tnext = 0
			for j in range(0, len(w)):
				tnext += (((w[j] - deltnext) * psip[j]) - (tp * psipp[j]))**2

			tnext = tnext**.5
			
			psinext = np.zeros_like(psip)
			for k in range(0, len(w)):
				psinext[k] = (tnext**-1) * (psip[k]*(w[k]-deltnext)-tp*psipp[k])
			
			return deltnext, tnext, psinext

		i = 0
		while i < self.nres:
			dnext, tnxt, psinext = compute_next(psicurr, psiprev, tprev, freqs)
			psiprev = psicurr
			psicurr = psinext
			tprev = tnxt
			E.append(dnext)
			t.append(tnxt)
			i+=1

		xE = E
		yEmin = upper - max(parms[1+self.nres:2*self.nres+1])
		yEmax = upper+.01

		self.name.calc_Q(parms, self.nres)

		if liveplot != True:
			plt.figure(figsize = (7,7))
			plt.vlines(xAvg, yEmin, yEmax, alpha = .45, linewidth = 3, color = 'b', label = r"$Average$")
			plt.vlines(xE, yEmin, yEmax, linestyle = 'dotted', label = r"$Base \, Energies$")
			plt.plot(xdata, ydata, 'o', ms = .3, label = r"$Raw \, Data$")
			plt.plot(xpeaks, yFit, linewidth = 4, alpha = .6, label = r"$Fitted \, Curve$")
			plt.plot(xpeaks, yuFit, alpha = .8, label = r"$Guesses$")
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			plt.title(title)
			plt.legend(loc = 0)
			plt.show()

		else:
			expt.plotter.plot_xy("Raw Data", xdata, ydata)
			expt.plotter.plot_xy("Guesses", xpeaks, yuFit)
			expt.plotter.plot_xy("Current Fit", xpeaks, yFit)
			expt.plotter.plot_y("Energies", E)
			expt.plotter.plot_y("Average", xAvg)

		print "E in GHz: ", np.divide(E, 1e9)
		print "t in MHz: ", np.divide(t, 1e6)
		return np.divide(E, 1e9), np.divide(t, 1e6), parms, chi

	def get_weighted_freqs(self, xdata, ydata, domain, upper = .99, liveplot = True, xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$"):
		x_coord, y_coord = self.name.find_peaks(xdata, ydata, domain)

		nres = len(x_coord)

		freqs = []

		i = 1
		while i <= 5:
			freqs.append(x_coord[i]-x_coord[i-1])
			i+=1

		gamma = .25*min(freqs)

		gammas = np.ones_like(x_coord)*gamma

		guesses = np.r_[upper, gammas, .99-y_coord, x_coord]

		popt, chi, yFit, yuFit, xpeaks = self.name.multi_lorentzian(xdata, 
			ydata, guesses, nres, domain, upper = guesses[0], plot = False, outside_use = True)

		xAvg = self.name.weighted_avg(popt, nres)

		yEmin = upper - max(popt[1+nres:2*nres+1])
		yEmax = upper+.01

		self.name.calc_Q(popt, nres)

		if liveplot != True:
			plt.figure(figsize = (7,7))
			plt.vlines(xAvg, yEmin, yEmax, alpha = .45, linewidth = 3, color = 'b', label = r"$Average$")
			plt.plot(xdata, ydata, 'o', ms = .3, label = r"$Raw \, Data$")
			plt.plot(xpeaks, yFit, linewidth = 4, alpha = .6, label = r"$Fitted \, Curve$")
			plt.plot(xpeaks, yuFit, alpha = .8, label = r"$Guesses$")
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			plt.legend(loc = 0)
			plt.show()

		else:
			expt.plotter.plot_xy("Raw Data", xdata, ydata)
			expt.plotter.plot_xy("Guesses", xpeaks, yuFit)
			expt.plotter.plot_xy("Current Fit", xpeaks, yFit)
			expt.plotter.plot_y("Average", xAvg)

		return popt, xAvg
















































