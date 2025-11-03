import pyvisa
import time
import numpy as np
import re

rm = pyvisa.ResourceManager()
instr = rm.open_resource("TCPIP0::192.168.6.150::inst0::INSTR")
instr.timeout = 10000

def check_response(label):
    r = instr.query("*OPC?")
    print(r)
    #assert r.strip() == "+1", f"{label} did not complete"
    print(f"[OK] {label}")
    print("ERRORS", instr.query("SYST:ERR?"))


def parse_sdata(s):
    f = np.fromstring(s, sep=',')
    return f[::2] + 1j*f[1::2]

def sweep(start, stop, points):
    
    print(instr.query("*IDN?").strip())
    instr.write('LSB;FMB') 
    instr.write('SENS1:FREQ:START ',start)
    instr.write('SENS1:FREQ:STOP ',stop)
    instr.write(':SENS1:SWE:POIN ',points)
    instr.write(':CALC1:PAR1:DEF S21')
    

    #instr.write('INIT:IMM; *WAI')
    instr.write(':SENS:HOLD:FUNC HOLD')
    instr.write(':TRIG:SING')
    # Query
    print("Querying...")
    sdata = instr.query_binary_values(':CALC1:DATA:SDAT?', datatype = 'd', container = np.array).reshape((-1,2))  # any data query
    sdata = sdata[:,0] + sdata[:,1]*1j

    
    print("Received response.")
    return sdata
    
   
    #af = parse_sdata(sdata)
    #print(af)

    instr.close()
    rm.close()
	
