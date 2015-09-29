import visa
import pandas as pd
import time

def main():
	rm = visa.ResourceManager()
	inst = rm.open_resource('TCPIP::192.168.1.214::inst0::INSTR')
	inst.timeout = 25000

	checksum = 'Agilent Technologies,E5071C,MY46524088,B.13.01\n'

	id_check = inst.query("*IDN?")

	if id_check != checksum:
		print "Error opening instrument"
		print id_check
		print checksum
		return "Error opening instrument"
	else:
		print "it worked"
		pass

	inst.write(":INIT1")
	print "initialized"
	time.sleep(.5)

	inst.write(":TRIG:SOUR")
	print "ready to start"
	time.sleep(.5)

	inst.write(":TRIG:SING")
	time.sleep(20)
	print inst.query("*OPC")

if __name__ == '__main__':
	main()