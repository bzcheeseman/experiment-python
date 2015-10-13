__author__ = "Aman LaChapelle"

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
		outputs = name.multi_lorentzian(parms = guesses, nres = nres, domain = domain, upper = upper, plot = False, outside_use = True)
		self.popt = outputs["popt"]
		self.chi = outputs["chi"]
		self.yFit = outputs["yFit"]
		self.yuFit = outputs["yuFit"]
		self.xpeaks = outputs["xpeaks"]

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

class get_twod_freqs(fitter):

	def __init__(self, path_to_data, datasets, guesses, Nx, Ny, domain, upper = None): 
		##for a more comprehensive instance of fitter use the actual file or edit it here
			##make sure the first dataset is the reflection measurement

		self.Nx = Nx
		self.Ny = Ny

		self.nres = Nx*Ny

		self.upper = upper
		self.domain = domain

		names = {}
		parameters = {}

		for i in range(0, Nx):
			name = fitter(path_to_data = path_to_data, dataset = datasets[i])
			names["(%d, %d)" % (1, i+1)] = name
			##now we have a dictionary of 'fitter' objects set up, 
				##we need the fit parameters stored in something similar
			parameters["(%d, %d)" % (1, i+1)] = name.pos_multi_lorentzian(parms = guesses, 
				nres = self.nres, domain = domain, upper = upper)

		##So initializing this class will fit all the peaks and store the fitter instances
			##and the parameters of each dataset in the set of datasets provided.

		self.names = names
		self.parameters = parameters
		

	def get_freqs(self, xlabel = r"$Frequency \, (Hz)$", ylabel = r"$Energy \, (dB)$", 
		title = r"$Spectrum \, and \, Fit$", disp_avg = True):

		##set up global variables here to get a list running
		base_freqs = {}
		phis = {}
		tunnelings = {}
		emmu = {}

		##find all the parameters
		for k in range(1, Ny+1): #indexing (?) - might cause an error
			##find parameters row by row in -|-|-|...-| shape
			for i in range(1, Nx+1):
				##define popt
				popt = parameters["(%d,%d)" % (k,i)]["popt"]

				phi = np.divide(np.sqrt(popt[1+self.nres:1+2*self.nres]), 
					np.linalg.norm(np.sqrt(popt[1+self.nres:1+2*self.nres])))

				mu = 0

				##find the mu
				for j in range(1, self.nres):
					mu += (popt[j+2*self.nres]*(phi[j-1]**2))

				base_freqs["(%d,%d)" % (k,i)] = mu
				emmu["(%d,%d)" % (k,i)] = np.subtract(popt[1+2*self.nres:], mu)
				phis["(%d,%d)" % (k,i)] = phi

				##now we have the phi and mu for (1, i+1), need to find t_(1,1)_(1,i+1)

				try: ##in case we're on the edge (i,j) = (1,j)
					t = 0
					for j in range(1, self.nres):
						t += (popt[j+2*self.nres]*phis["(%d,%d)" % (k,i-1)][j-1]\
							*phis["(%d,%d)" % (k,i)][j-1])

					tunnelings["(%d,%d)-(%d,%d)" % (k,i-1, k,i)] = t

					##so now we have the mu/t/phi for the first row (hopefully)
						##time to find the vertical tunnelings

					try: ##in case we're on the edge (i,j) = (i, Ny)
						t = 0
						t = -np.sqrt(np.sum( (emmu["(%d,%d)" % (k,i-1)]*phis["(%d,%d)" % (k,i)] \
							- tunnelings["(%d,%d)-(%d,%d)" % (k,i-1, k,i)]*phis["(%d,%d)" % (k,i)])**2 ))
						tunnelings["(%d,%d)-(%d,%d)" % (k+1, i)] = t

					except IndexError:
						tunnelings["(%d,%d)-(%d,%d)" % (k+1, i)] = 0

				except IndexError:
					tunnelings["(%d,%d)-(%d,%d)" % (k,i-1, k,i)] = 0

				##aaaaand carriage return, next row, same pattern

				##hopefully all the extraneous tunnelings (to the left and below) will be zero
					##and will be easily filtered out

		return {"base_freqs":base_freqs, "phis":phis, "tunnelings":tunnelings}













