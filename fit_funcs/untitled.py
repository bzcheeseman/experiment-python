import numpy as np

def main():
	Nx = 3
	Ny = 4
	nres = Nx*Ny
	ones = np.ones(nres)

	base_freqs = {}
	phis = {}
	tunnelings = {}
	emmu = {}

	##find all the parameters
	for k in range(1, Ny+1): #indexing (?) - might cause an error
		##find parameters row by row in -|-|-|...-| shape
		for i in range(1, Nx+1):
			##define popt
			popt = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
			1, 1, 1, 0.043386, 0.120982, 0.0452141, 0.130105, 0.1574, 0.00154957, 
			0.0105796, 0.230829, 0.0470029, 0.0727523, 0.101828, 0.0383733,
			0.0541386, 0.130268, -0.0147677, 0.15101, -0.0279006, -0.0104177, 
			0.0225727, -0.00332945, -0.0890986, -0.000705812, -0.155565, 
			-0.0562047, 0.0490607, 0.0889439, 0.0193588, 0.146088, 0.027647, 
			-0.0621784, 0.0511878, 0.00294847, -0.120719, 0.000575399, -0.15187, 
			-0.05104251, 0.6239, 10.6239, 10.4156, 10.3547, 10.1686, 10.1531, 10.0491, 9.94816, 
			9.89371, 9.82277, 9.68104, 9.57655, 9.37879]

			phi = popt[1+i*nres:1+i*nres+12]

			##find the mu
			mu = 0
			for a in range(nres, 0, -1):
				for l in range(0, nres):
					mu += (popt[-a]*np.absolute(phi[l]**2))

			print mu
		
			base_freqs["(%d,%d)" % (k,i)] = mu
			emmu["(%d,%d)" % (k,i)] = np.subtract(popt[-nres:], mu)
			phis["(%d,%d)" % (k,i)] = phi

			##now we have the phi and mu for (1, i+1), need to find t_(1,1)_(1,i+1)

			try: ##in case we're on the edge (i,j) = (1,j)
				t = 0
				for j in range(1, nres):
					t += (popt[j+2*nres]*phis["(%d,%d)" % (k,i-1)][j-1]\
						*phis["(%d,%d)" % (k,i)][j-1])

				tunnelings["(%d,%d)-(%d,%d)" % (k,i-1, k,i)] = t

				##so now we have the mu/t/phi for the first row (hopefully)
					##time to find the vertical tunnelings

				try: ##in case we're on the edge (i,j) = (i, Ny)
					t = 0
					t = -np.sqrt(np.sum( (emmu["(%d,%d)" % (k,i-1)]*phis["(%d,%d)" % (k,i)] \
						- np.multiply(tunnelings["(%d,%d)-(%d,%d)" % (k,i-1, k,i)],phis["(%d,%d)" % (k,i)])**2 )))
					tunnelings["(%d,%d)-(%d,%d)" % (k+1,i, k,i)] = t

				except KeyError:
					tunnelings["(%d,%d)-(%d,%d)" % (k+1, i)] = 0

			except KeyError:
				tunnelings["(%d,%d)-(%d,%d)" % (k,i-1, k,i)] = 0

			##aaaaand carriage return, next row, same pattern

			##hopefully all the extraneous tunnelings (to the left and below) will be zero
				##and will be easily filtered out

	return {"base_freqs":base_freqs, "phis":phis, "tunnelings":tunnelings}

if __name__ == '__main__':
	main()