def error_code(code, debug = False):

	import os

	path_to_errtext, _ = os.path.split(os.path.realpath(__file__))

	os.chdir(path_to_errtext)

	if debug == True:
		with open("error_codes.txt", 'r') as f:
			for line in f.readlines():
				print line
		f.close()

	with open("error_codes.txt", 'r') as f:
		lines = f.readlines()
		try:
			print lines[code]
			f.close()
			return code

		except IndexError:
			print "Code not found, suggest the code and text for addition to the text?"
			f.close()
			return 8


if __name__ == "__main__":
	code = int(raw_input("Code to test: "))
	error_code(code, debug = True)