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

	def lorentzian():
		return 0


if __name__ == "__main__":
	#run with debug and some sample data