import numpy as np
from fitter import *
import matplotlib.pyplot as plt

class get_freqs(fitter):

	def __init__(self, name, path_to_data, dataset, guesses, nres, domain, upper = None): 
		## for a more comprehensive instance of fitter use the actual file or edit it here
		name = fitter(path_to_data = path_to_data, dataset = dataset)
		
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

	def get_freqs(self, xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$"):
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
		yEmax = self.upper

		plt.figure(figsize = (7,7))
		plt.vlines(xE, yEmin, yEmax, linestyle = 'dashed', label = r"$Base \, Energies$")
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
		return np.divide(E, 1e9), np.divide(t, 1e6)





