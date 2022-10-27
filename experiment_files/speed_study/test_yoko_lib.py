import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import time
import datetime
from glob2 import glob

from importlib import reload
from instruments import instek3032
from instruments import yoko7651
from instruments import zvk
from instruments import fsva

reload(yoko7651)
yoko = yoko7651.Yoko7651('GPIB1::3::INSTR')

time.sleep(10)
# testing program creation

voltage_values = [1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
intervals = [1, 5]
sweeps = [0, 2]
input("Press enter to begin programming.")
with yoko.write_program() as program : 
    input("Press enter to set mode and range.")
    program.source_voltage()
    program.range_voltage(12)
    input("Press enter to input program values.")
    for v in voltage_values:
        time.sleep(0.5)
        program.voltage(v)
    
input("Finished writing. Press Enter to continue.")
for i, interval in enumerate(intervals) : 
    input("Press enter to set sweep and interval")
    time.sleep(1)
    yoko.interval(interval)
    time.sleep(1)
    yoko.sweep_duration(sweeps[i])
    
    input(f"Press Enter to run program : sweep {sweeps[i]:.1f} s, interval {interval:.1f} s")
    
    yoko.run_program()
    time.sleep(7.5)
    
    yoko.hold_program()
    input("Program halted. Press Enter to resume")
    yoko.resume_program()
    time.sleep(10)
    
print("Finished.")
del yoko