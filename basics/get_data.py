__author__ = "Aman LaChapelle"

from error_handling import *
import os

cwd, _ = os.path.split(os.path.realpath(__file__))

def get_data(path_to_data, dataset, datatype = "csv", h5labels = [None, None, None], path_to_library = cwd+"/.."):

	h5xlabel = h5labels[0]
	h5ylabel = h5labels[1]
	h5zlabel = h5labels[2]

	if cwd == path_to_data:
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

			error_code(0)
			
			return data
		


		else:
			f1 = h5.File(dataset, 'r')
			xdata = np.array(f1['h5xlabel'])
			ydata = np.array(f1['h5ylabel'])

			data = [xdata, ydata]
 
			error_code(0)

			return data

	elif h5xlabel == None or h5ylabel == None:
		error_code(5)
		return 5

if __name__ == "__main__":
	print "Usage:\nget_data(path_to_data, dataset, **kwargs)"
	print "Currently accepts csv files and h5 files of 2 or 3 dimensions"


