import pyvisa
import time
import numpy as np


rm = pyvisa.ResourceManager()
instr = rm.open_resource("TCPIP0::192.168.6.150::inst0::INSTR")
instr.timeout = 10000

def check_response(label):
    r = instr.query("*OPC?")
    assert r.strip() == "+1", f"{label} did not complete"
    print(f"[OK] {label}")
    print("ERRORS", instr.query("SYST:ERR?"))


def parse_sdata(s):
    f = np.fromstring(s, sep=',')
    return f[::2] + 1j*f[1::2]

if __name__ == "__main__":
    print(instr.query("*IDN?").strip())

    instr.write("*RST")
    instr.write("*CLS")
    check_response("*RST, *CLS")


    instr.write("CALC:PAR:DEL:ALL")
    check_response("CALC:PAR:DEL:ALL")
    instr.write("DISP:WIND1:TRAC1:DEL")
    check_response("DISP:WIND1:TRAC1:DEL")

    # select s21
    instr.write('CALC:PAR:DEF "MyMeas",S21')
    instr.write('CALC:PAR:SEL "MyMeas"')
    check_response("use S21")

    # display
    instr.write('DISP:WIND1:TRAC1:FEED "MyMeas"')

    # sweep
    instr.write("SENS:FREQ:START 25e9")
    instr.write("SENS:FREQ:STOP 30e9")
    instr.write("SENS:SWE:POIN 10")
    check_response("Sweep settings")

    instr.write("INIT:CONT OFF")
    instr.write("FORM ASC")

    instr.write("INIT:IMM")
    check_response("Sweep")

    # Query
    print("Querying...")
    sdata = instr.query("CALC:DATA? SDATA")  # any data query
    print("Received response.")

    af = parse_sdata(sdata)
    print(af)

    instr.close()
    rm.close()