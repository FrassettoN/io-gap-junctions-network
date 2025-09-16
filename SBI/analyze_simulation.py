from scipy.signal import find_peaks
import numpy as np
import scipy.signal as signal


def compute_firing_rate(spike_times, milliseconds):
    seconds = milliseconds / 1000.0
    return len(spike_times) / seconds


def compute_sto(v_m, t, spike_times=None, f_range=(5, 12), power_threshold=1e-6):
    """
    Compute subthreshold oscillations (STO) in a given frequency range.
    Returns dominant frequency and amplitude, or None if no oscillation is present.
    """
    vm_clean = v_m.copy()

    dt_ms = np.mean(np.diff(t))  # robust dt in ms
    fs = 1000.0 / dt_ms  # Hz (samples per second)

    # choose nperseg no larger than signal length (and power-of-two often ok)
    nperseg = min(2048, len(vm_clean))
    frequencies, power = signal.welch(vm_clean, fs=fs, nperseg=len(vm_clean))

    # Focus on STO frequency range (1–100 Hz)
    sto_band = (frequencies >= 1) & (frequencies <= 15)
    sto_freq = frequencies[sto_band][np.argmax(power[sto_band])]

    if sto_freq - 1 > 1e-2:
        # Find peaks and troughs
        peaks, _ = find_peaks(vm_clean, distance=400)
        troughs, _ = find_peaks(-vm_clean, distance=400)

        # f Make sure we align peaks/troughs
        min_len = min(len(peaks), len(troughs))
        if min_len > 0:
            amplitudes = np.abs(vm_clean[peaks[:min_len]] - vm_clean[troughs[:min_len]])
            mean_amp = np.mean(amplitudes)
            amp_variance = np.var(amplitudes)  # Calculate variance of amplitudes
            amp_std = np.std(amplitudes)  # Standard deviation for easier interpretation
        else:
            mean_amp = 0
            amp_variance = 0
            amp_std = 0
    else:
        mean_amp = 0
        amp_variance = 0
        amp_std = 0

    return sto_freq, mean_amp, amp_std


def analyze(vm, sr, milliseconds):
    vm_values = vm.events["V_m"]
    times = vm.events["times"]

    vm_mean = float(np.mean(vm_values))

    sr_spike_times = sr.events["times"]
    firing_rate = compute_firing_rate(sr_spike_times, milliseconds)

    sto_freq, sto_amp, sto_std = compute_sto(vm_values, times, sr_spike_times)

    return [
        round(firing_rate, 2),
        round(sto_freq, 2),
        round(sto_amp, 2),
        round(sto_std, 2),
    ]
