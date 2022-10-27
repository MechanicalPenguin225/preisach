### may 23 2022, Lou Bernabeu
### ----------------- IMPORTS
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

V_POINTS = 11
RAMP_DURATIONS = 0.1*np.arange(1, 31)
N_FREQ_SWEEP_POINTS = 101 # number of steps for VNA freq sweeps
VNA_WINDOW_WIDTH = 200e6
TRACENAME = "CH1DATA"
FILENAME = r'\speed_sweeps'
V_MIN = 0 
V_MAX = 10

# Inferring some variables from parameters
increasing_sweep = np.linspace(V_MIN, V_MAX, V_POINTS)
decreasing_sweep = np.flip(increasing_sweep)[1:]
V_values = np.concatenate((increasing_sweep, decreasing_sweep))
dV = V_values[1] - V_values[0]
sweep_speeds = dV/RAMP_DURATIONS
n_speeds = len(RAMP_DURATIONS)

N_total_points = len(RAMP_DURATIONS)*2*V_POINTS

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
def ramp_yoko(voltage, sweep_speed):
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
        time.sleep(delta_t + 0.1) # gotta go fast so I'm not waiting a lot.

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
zvk_freqs = np.zeros((N_total_points, N_FREQ_SWEEP_POINTS))
zvk_traces = np.zeros((N_total_points, N_FREQ_SWEEP_POINTS), dtype = "complex")

# initializing setup at 0V

filename_final = input(f'Filename ? (or Enter to cancel, or Space+Enter for {FILENAME}): ')
if filename_final == '':
    del vna
    del yoko
    print("measurement cancelled.")
    quit()

filename_final = FILENAME + when.strftime('%Y%m%d-%Hh%Mmin%Ss')
t0 = time.time()
ramp_yoko(0, 0.1)

for i, speed in enumerate(sweep_speeds):

    for j, V in enumerate(V_values): 

        point_linear_index = i*(2*V_POINTS - 1) + j

        freq_limits = freq_window(V, VNA_WINDOW_WIDTH)
        vna.freq_start_stop = freq_limits
        ramp_yoko(V, speed)
        vna.trigger()

        while vna.busy() :
            time.sleep(0.01)

        f, z = vna.get_data(trace = TRACENAME)

        zvk_traces[point_linear_index, :] = z
        zvk_freqs[point_linear_index, :] = f

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
    'datetime': when


}

# -------------------- DOING UNNAMEABLE STUFF TO THE DATA (I might get sent to The Hage for this)
zvk_traces_dB = 20*np.log10(np.abs(zvk_traces))

band_freqs, max_values = get_band_pass_position(zvk_traces_dB, zvk_freqs)

np.savez(PATH + filename_final, zvk_traces = zvk_traces, zvk_freqs = zvk_freqs, V_values = V_values, sweep_speeds = sweep_speeds, meta = meta, comment = comment)
t1 = time.time()

print(f"Measurement + save lasted {(t1 - t0):.1f} s")

# -------------------- PLOTTING
colors = plt.cm.viridis(np.linspace(0, 1, n_speeds))

center_freqs = np.mean(band_freqs, axis = -1)*1e-9

fig, ax = plt.subplots(1, 2, figsize = (12, 5))
ax_l, ax_r = ax

ax_r.grid(True)
ax_r.axhline(color = 'k', lw = 0.5)

for axis in ax:
    axis.set_xlabel("V  (V)")
    axis.set_prop_cycle('color', list(colors))

for i, speed in enumerate(sweep_speeds):
    min_index = i*(2*V_POINTS - 1)
    max_index = min_index + 2*V_POINTS - 1

    cycle = V_values[min_index:max_index + 1]

    print(min_index, max_index)

    label = f"v = {speed:.1f} V/s"

    ax_l.plot(V_values, center_freqs[min_index:max_index], label = label)
    ax_r.plot(V_values, center_freqs[min_index:max_index] - theoretical_peak_position(V_values)*1e-9, label = label)

ax_l.legend()


fig.savefig(PATH + filename_final + ".png")

plt.show()

# -------------------- CLEANING UP

del vna
del yoko