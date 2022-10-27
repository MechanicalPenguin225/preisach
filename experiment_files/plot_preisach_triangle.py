### may 23 2022, Lou Bernabeu
### ----------------- IMPORTS
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import time
from glob2 import glob



# -------------------- PARAMETERS
PATH =r"C:\Users\etudiant.jeip\Documents\tunable-filter\preisach"
FILE = r"\preisach_triangle20220719-17h41min46s.npz" #r"\preisach_triangle20220525-14h44min56s.npz"

npz_file = dict(np.load(PATH+FILE, allow_pickle = True))

zvk_traces = npz_file["zvk_traces"]
zvk_freqs = npz_file["zvk_freqs"]
V_values = npz_file["V_values"]
indices = npz_file["indices"]
meta = npz_file["meta"].item()
N_VOLTAGE_POINTS = meta["voltage_points"]
SWEEP_SPEED = meta["sweep_speed"]
# -------------------- FUNCTIONS


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

# -------------------- DOING UNNAMEABLE STUFF TO THE DATA (I might get sent to The Hage for this)
zvk_traces_dB = 20*np.log10(np.abs(zvk_traces))

band_freqs, max_values = get_band_pass_position(zvk_traces_dB, zvk_freqs)

# -------------------- PLOTTING
V_positions_beta = V_values[indices[:, 1]]
V_positions_alpha = V_values[indices[:, 0]]

center_freqs = np.mean(band_freqs, axis = -1)*1e-9

fig, ax = plt.subplots(1, 3, figsize = (20, 5))
ax_l, ax_m, ax_r = ax


for i in range(N_VOLTAGE_POINTS):
    j = i + 1
    min_index = j*(j - 1)//2
    max_index = j*(j + 1)//2

    V_values = V_positions_beta[min_index:max_index]
    print(min_index, max_index)
    ax_m.plot(V_values, center_freqs[min_index:max_index] - theoretical_peak_position(V_values)*1e-9)
    ax_l.plot(V_values, center_freqs[min_index:max_index])

ax_m.set_xlabel(r"$V$ (V)")
ax_m.set_ylabel(r"$f$ (GHz)")

ax_l.set_xlabel(r"$V$ (V)")
ax_l.set_ylabel(r"$f$ (GHz)")

im = ax_r.tripcolor(V_positions_beta, V_positions_alpha, center_freqs)
ax_r.set_xlabel(r"$\beta$")
ax_r.set_ylabel(r"$\alpha$")
fig.colorbar(im, ax = ax_r)

fig.suptitle(f"Voltage sweep speed = {SWEEP_SPEED:.2f} V/s")

titles = ["Whole hysteresis cycle", "Deviation from linear expression", r"Integrals of $\mu(\alpha, \beta)$"]

for i, axis in enumerate(ax) : 
    axis.set_title(titles[i])

plt.show()
fig.savefig("plot_" + FILE[1:-5] + ".png")

