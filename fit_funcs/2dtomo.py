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

	#1,1, 2,2, 3,3, ... 1,2, 2,3, 3,4, 4,5, 

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

####find mu 11
	mu11 = 0
	for i in range(0, nres):
		mu11 += meas_energies[i]*(np.absolute(meas_ref_phi[i]))

	base_freqs["(1,1)"] = mu11
	print base_freqs

####find t_11->12###not working
	t1112 = 0
	for i in range(0, nres):
		t1112 += meas_energies[i] * np.absolute(meas_trans_phi[i])

	tunnelings["(1,1)-(1,2)"] = t1112
	print tunnelings

if __name__ == '__main__':
	main()