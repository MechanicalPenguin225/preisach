### may 23 2022, Lou Bernabeu
### ----------------- IMPORTS
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import time
import datetime
from glob2 import glob

from importlib import reload
from instruments import yoko7651
from instruments import zvk

reload(zvk)
reload(yoko7651)
CURRENT_YOKO_VOLTAGE = 0
when = datetime.datetime.now()


# -------------------- PARAMETERS
PATH = r"S:\analysis\python\notebooks\tunable-filter\preisach\memory_study"

N_VOLTAGE_POINTS = 11 # number of subdivisions of the [0, 10V] interval for approximating the weight function
SMALLEST_STEP_RAMP_TIME = 0.1 # Yoko only allows sweeps to last for 0.1 to 3600 s, with a resolution of 0.1s . 
N_FREQ_SWEEP_POINTS = 51 # number of steps for VNA freq sweeps
VNA_WINDOW_WIDTH = 100e6
DELAYS = np.logspace(-6, 1, 15)
TRACENAME = "CH1DATA"
FILENAME = r'\memory_sweeps'


# Inferring some variables from parameters
V_values = np.linspace(0, 10, N_VOLTAGE_POINTS) 

N_total_points = N_VOLTAGE_POINTS*(N_VOLTAGE_POINTS - 1) 
N_delay_values = len(DELAYS)
sweep_speed = (V_values[1] - V_values[0])/SMALLEST_STEP_RAMP_TIME # speed of sweep in V/s

print(f"SWEEP SPEED IS {sweep_speed:.2f} V/s")

# -------------------- INSTRUMENT SETUP
vna = zvk.Zvk("GPIB1::20::INSTR")
yoko = yoko7651.Yoko7651("GPIB1::1::INSTR")

vna.sweep_points = N_FREQ_SWEEP_POINTS
vna.power = -12 #dBm
vna.averaging = 0
vna.sweep_count = 1
VNA_SWEEP_DUR = 1.1*vna.sweep_duration
vna.single_sweep = True

# -------------------- FUNCTIONS
def ramp_yoko(voltage, delay):
    global CURRENT_YOKO_VOLTAGE
    delta_v = abs(voltage - CURRENT_YOKO_VOLTAGE)
    if delta_v != 0 :
        delta_t = round(delta_v/sweep_speed, 1)
        print(f"GOING TO VOLTAGE {voltage:.1f}, deltaV = {delta_v:.1f} V, delta_t = {delta_t:.1f} s")
        yoko.interval(delta_t)
        yoko.sweep_duration(delta_t)

        with yoko.write_program() as program : 
            program.source_voltage()
            program.range_voltage(12)
            program.voltage(voltage)
        
        yoko.run_program()
        time.sleep(delta_t + delay)

    CURRENT_YOKO_VOLTAGE = voltage
    return None

def get_max_peaks(amps, f_list):
    if f_list.ndim == 1 : 
        v_shape, _ = amps.shape
        max_freqs = np.zeros(v_shape)
        max_values = np.zeros(v_shape)
        for i in range(v_shape) :
            v_slice = amps[i, :]
            max_freqs[i] = f_list[np.argmax(v_slice)]
            max_values[i] = np.max(v_slice)
        return max_freqs, max_values
    
    elif f_list.ndim == 2:
        v_shape, _ = amps.shape
        max_freqs = np.zeros(v_shape)
        max_values = np.zeros(v_shape)
        for i in range(v_shape) :
            v_slice = amps[i, :]
            max_freqs[i] = f_list[i, np.argmax(v_slice)]
            max_values[i] = np.max(v_slice)
        return max_freqs, max_values
    else : raise ValueError("f_list must have one or two dims.")

def get_band_pass_position(amps_dB, f_list, cutoff = 3):
    """ Gives passing band properties for a set of VNA traces.
    
    ---------
    Arguments : 
    
    amps_dB : np.ndarray, 2-dimensional
        Amplitudes in dB of the VNA trace
        first dimension is voltage index, second dimension is trace frequency.
        ie. amps_dB[2, :] is the third trace.
    
    f_list : np.ndarray, 1 or 2-D
        if 1D : the common list of freqs for all VNA traces.
        if 2D : the list of the respective freq lists for each different VNA trace.
        
    cutoff : positive float
        when to cutoff for chossing the pass band. 3 gives the 3dB pass band, for example."""
    
    if f_list.ndim == 1 : 
        v_shape, _ = amps_dB.shape
        band_freqs = np.zeros((v_shape, 2))
        max_values = np.zeros(v_shape)
        for i in range(v_shape) :
            v_slice = amps_dB[i, :]
            max_val = np.max(v_slice)
            max_values[i] = max_val
            pass_band = np.nonzero(v_slice >= (max_val - cutoff)) # get indices of values that are less that 3dB from plateau value
            band_freqs[i, :] = f_list[np.min(pass_band)], f_list[np.max(pass_band)]
        return band_freqs, max_values
    
    elif f_list.ndim == 2:
        v_shape, _ = amps_dB.shape
        band_freqs = np.zeros((v_shape, 2))
        max_values = np.zeros(v_shape)
        for i in range(v_shape) :
            v_slice = amps_dB[i, :]
            max_val = np.max(v_slice)
            max_values[i] = max_val
            pass_band = np.nonzero(v_slice >= (max_val - cutoff)) # get indices of values that are less that 3dB from plateau value
            band_freqs[i, :] = f_list[i, np.min(pass_band)], f_list[i, np.max(pass_band)]
        return band_freqs, max_values
    else : raise ValueError("f_list must have one or two dims.")
    
def get_band_pass_single_trace(amps_dB, f_list, cutoff = 3):
    """Returns 3dB passing band limits + height of max of peak for a single VNA trace."""
    max_val = np.max(amps_dB)
    pass_band = np.nonzero(amps_dB >= (max_val - cutoff)) # get indices of values that are less that 3dB from plateau value
    band_freqs = f_list[np.min(pass_band)], f_list[np.max(pass_band)]
    return band_freqs, max_val

def theoretical_peak_position(V):
    return (38e9/10)*V + 2e9

def freq_window(V, delta):
    """Gives the freq window to give  to the VNA for input voltage V and window width delta (delta in Hz)"""
    if V <= (10/38e9)*(40e9 - 2e9 - delta/2) :
        return np.array([-delta/2, delta/2]) + (38e9/10)*V + 2e9
    else : 
        return np.array([40e9 - delta, 40e9])
# -------------------- CODE

# initializing data arrays
voltages = np.zeros((N_total_points))
zvk_freqs = np.zeros((N_total_points, N_FREQ_SWEEP_POINTS))
zvk_traces = np.zeros((N_delay_values, N_total_points, N_FREQ_SWEEP_POINTS), dtype = "complex")

# initializing setup at 0V

filename_final = input(f'Filename ? (or Enter to cancel, or Space+Enter for {FILENAME}): ')
if filename_final == '':
    del vna
    del yoko
    print("measurement cancelled.")
    quit()

filename_final = FILENAME + when.strftime('%Y%m%d-%Hh%Mmin%Ss')
t0 = time.time()
ramp_yoko(0, 0)

for k, delay in enumerate(DELAYS):

    print(f"DOING DELAY = {delay:.1e} s")

    for i in range(N_VOLTAGE_POINTS - 1):
        cycle_number = i + 1
        print(i, cycle_number*(cycle_number - 1))
        print(voltages)
        for j in range(cycle_number): # ramping all the way up to cycle_number-1
            point_linear_index = cycle_number*(cycle_number - 1) + j

            V = V_values[j]

            freq_limits = freq_window(V, VNA_WINDOW_WIDTH)
            vna.freq_start_stop = freq_limits

            ramp_yoko(V, delay)

            vna.trigger()

            while vna.busy() :
                time.sleep(0.01)

            f, z = vna.get_data(trace = TRACENAME)

            zvk_traces[k, point_linear_index, :] = z
            zvk_freqs[point_linear_index, :] = f
            voltages[point_linear_index] = V
        
        for j in range(cycle_number, 0, -1): # loop from i all the way down to 0 (excluding 0 cause it'll be taken care of in the next loop)
            point_linear_index = cycle_number*(cycle_number - 1) + cycle_number + (cycle_number - j)

            V = V_values[j]

            freq_limits = freq_window(V, VNA_WINDOW_WIDTH)
            vna.freq_start_stop = freq_limits

            ramp_yoko(V, delay)

            vna.trigger()

            while vna.busy() :
                time.sleep(0.01)

            f, z = vna.get_data(trace = TRACENAME)

            zvk_traces[k, point_linear_index, :] = z
            zvk_freqs[point_linear_index, :] = f
            voltages[point_linear_index] = V

# -------------------- SAVING DATA AS NPY FILE

comment = 'Test run'
S_parameter = 'S21'
ch = 1

meta = {
    'nb_points': vna.sweep_points,
    'sweep_count': vna.sweep_count,
    "sweep_duration":vna.sweep_duration,
    "averaging": vna.averaging,
    'VBW': vna.bandwidth,
    'power': vna.power,
    'S_parameter': S_parameter,
    'datetime': when,
    'voltage_points': N_VOLTAGE_POINTS,
    'sweep_speed': sweep_speed,


}

# -------------------- DOING UNNAMEABLE STUFF TO THE DATA (I might get sent to The Hage for this)
zvk_traces_dB = 20*np.log10(np.abs(zvk_traces))

band_freqs = np.array([get_band_pass_position(zvk_traces_dB[k, :, :], zvk_freqs)[0] for k in range(N_delay_values)])

np.savez(PATH + filename_final, zvk_traces = zvk_traces, zvk_freqs = zvk_freqs, V_values = V_values, voltage_path = voltages, delays = DELAYS, meta = meta, comment = comment)
t1 = time.time()

print(f"Measurement + save lasted {(t1 - t0):.1f} s")

# -------------------- PLOTTING
colors = plt.cm.viridis(np.linspace(0, 1, N_delay_values))


center_freqs = np.mean(band_freqs, axis = -1)*1e-9

fig, ax = plt.subplots(figsize = (20, 5))
ax.set_prop_cycle('color', list(colors))

ax.axhline(color = 'k', lw = 0.5)
ax.grid(True)

theory = theoretical_peak_position(voltages)*1e-9

for k in range(N_delay_values):
    ax.plot(voltages, center_freqs[k, :] - theory, label = f"delay = {DELAYS[k]:.1e} s")
    
ax.set_xlabel("V (V)")
ax.set_ylabel(r"$\Delta f$ (GHz)")

plt.show()

# -------------------- CLEANING UP

del vna
del yoko