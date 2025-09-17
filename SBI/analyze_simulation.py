from scipy.signal import find_peaks
import numpy as np
import scipy.signal as signal


def compute_firing_rate(spike_times, milliseconds):
    seconds = milliseconds / 1000.0
    return len(spike_times) / seconds


def compute_sto(v_m, t, milliseconds, V_th, spike_times=None, min_amplitude=1.5):
    """
    Compute subthreshold oscillations (STO) in a given frequency range.
    Returns dominant frequency and amplitude, or None if no oscillation is present.
    """
    vm_clean = v_m.copy()

    peaks, _ = find_peaks(vm_clean, height=(None, V_th))
    peaks_values = vm_clean[peaks]

    troughs, _ = find_peaks(-vm_clean)
    troughs_values = vm_clean[troughs]

    min_len = min(len(peaks), len(troughs))
    if min_len > 0:
        amplitudes = np.abs(peaks_values[:min_len] - troughs_values[:min_len])

        valid_indices = amplitudes >= min_amplitude

        if np.any(valid_indices):
            valid_amplitudes = amplitudes[valid_indices]
            valid_peaks = peaks[:min_len][valid_indices]
            valid_troughs = troughs[:min_len][valid_indices]

            seconds = milliseconds / 1000.0
            sto_freq = len(valid_amplitudes) / seconds
            mean_amp = np.mean(valid_amplitudes)
            amp_std = np.std(valid_amplitudes)
        else:
            sto_freq = 0
            mean_amp = 0
            amp_std = 0
    else:
        sto_freq = 0
        mean_amp = 0
        amp_std = 0

    return sto_freq, mean_amp, amp_std


def analyze(vm, sr, milliseconds, V_th):
    vm_values = vm.events["V_m"]
    times = vm.events["times"]

    vm_mean = float(np.mean(vm_values))

    sr_spike_times = sr.events["times"]
    firing_rate = compute_firing_rate(sr_spike_times, milliseconds)

    sto_freq, sto_amp, sto_std = compute_sto(
        vm_values, times, milliseconds, V_th, sr_spike_times
    )

    return [
        round(firing_rate, 2),
        round(sto_freq, 2),
        round(sto_amp, 2),
        round(sto_std, 2),
    ]
