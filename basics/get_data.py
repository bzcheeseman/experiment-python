from error_handling import *
import os

cwd = os.getcwd()

def get_data(path_to_data, dataset, datatype = "csv", h5xlabel = None, h5ylabel = None, h5zlabel = None, 
	path_to_errtext = cwd+"/..", path_to_library = cwd+"/.."):
	import os

	err = error_handling(path_to_errtext, path_to_library)

	current = os.getcwd()

	if current == path_to_data:
		pass

	else:
		os.chdir(path_to_data)

	if datatype == "csv":
		import pandas as pd
		import numpy as np
		data = np.array(pd.read_csv("%s.csv" % dataset, header = 2))
		return data

	if datatype == "h5" and h5xlabel != None and h5ylabel != None:
		import h5py as h5

		if h5zlabel != None:
			f1 = h5.File(dataset, 'r')
			xdata = np.array(f1['h5xlabel'])
			ydata = np.array(f1['h5ylabel'])
			zdata = np.array(f1['h5zlabel'])

			data = [xdata, ydata, zdata]

			err.error_code(0)
			
			return data
		


		else:
			f1 = h5.File(dataset, 'r')
			xdata = np.array(f1['h5xlabel'])
			ydata = np.array(f1['h5ylabel'])

			data = [xdata, ydata]
 
			err.error_code(0)

			return data

	elif h5xlabel == None or h5ylabel == None:
		err.error_code(5)
		return 5

if __name__ == "__main__":
	print "useage"


