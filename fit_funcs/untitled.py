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

	parms = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
			1, 1, 1, 0.0306307, 0.0580278, 0.0748212, 0.146737, 0.181921, 0.0281016, 
			0.034514, 0.200078, 0.0371738, 0.117055, 0.0528912, 0.0380498, 
			0.043147, 0.0949833, 0.00628087, 0.147535, -0.0106582, -0.038059, 
			0.0620694, -0.0396205, -0.0854393, -0.0280026, -0.0994017, 
			-0.0528338, 0.0478125, 0.134955, -0.00527789, 0.110207, 0.0112502, 
			-0.0561601, 0.0539967, 0.030345, -0.132117, 0.0166768, -0.157278, 
			-0.0544092, 10.5957, 10.3982, 10.3218, 10.149, 10.1313, 10.0355, 9.95991, 
			9.87136, 9.83897, 9.6637, 9.58779, 9.39761]

	meas_ref_phi = parms[1+nres:1+2*nres]
	meas_energies = parms[-nres:]
	meas_trans_phi = parms[1+2*nres:-nres]

	mu11 = 0
	for i in range(0, nres):
		mu11 += meas_energies[i]*(meas_ref_phi[i]**2)

	base_freqs["(1,1)"] = mu11
	print mu11


	'''
	##find all the parameters
	for k in range(1, Ny+1): #indexing (?) - might cause an error
		##find parameters row by row in -|-|-|...-| shape
		for i in range(1, Nx+1):
			##define popt

			popt = parms

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

	return {"base_freqs":base_freqs, "phis":phis, "tunnelings":tunnelings}'''

if __name__ == '__main__':
	main()