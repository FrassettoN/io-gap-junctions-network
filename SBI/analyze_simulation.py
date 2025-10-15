from scipy.signal import find_peaks
import numpy as np
import scipy.signal as signal


def compute_firing_rate(spike_times, milliseconds):
    seconds = milliseconds / 1000.0

    return len(spike_times) / seconds


def compute_isi(spike_times):
    if len(spike_times) < 2:
        return 0.0

    # Calculate intervals between consecutive spikes
    intervals = np.diff(spike_times)

    mean_isi = np.mean(intervals)

    return mean_isi


def find_spike_peaks(v_m, t, spike_times):
    peaks, _ = find_peaks(v_m)
    spike_peaks = []

    if spike_times is not None and len(spike_times) > 0 and len(peaks) > 0:
        for target_time in spike_times:
            # Find the time index closest to target_time
            time_distances = np.abs(t - target_time)
            target_idx = np.argmin(time_distances)

            # Find the peak nearest to this time index
            peak_distances = np.abs(peaks - target_idx)
            nearest_peak_idx = np.argmin(peak_distances)
            spike_peaks.append(peaks[nearest_peak_idx])

        spike_peaks = np.array(spike_peaks)

    return spike_peaks


def get_inter_spike_intervals(v_m, t, spike_times):
    intervals = []
    spike_peaks = find_spike_peaks(v_m, t, spike_times)

    if len(spike_peaks) > 0:
        # Interval from start to first spike
        intervals.append((0, spike_peaks[0]))

        # Intervals between consecutive spikes
        for i in range(len(spike_peaks) - 1):
            segment = v_m[spike_peaks[i] :]
            troughs, _ = find_peaks(-segment)
            if len(troughs) > 0:
                # Convert relative index back to absolute index
                interval_start = spike_peaks[i] + troughs[0]
            else:
                interval_start = spike_peaks[i]
            intervals.append((interval_start, spike_peaks[i + 1]))

        # Interval from last spike to end
        segment = v_m[spike_peaks[-1] :]
        troughs, _ = find_peaks(-segment)
        if len(troughs) > 0:
            interval_start = spike_peaks[-1] + troughs[0]
        else:
            interval_start = spike_peaks[-1]
        intervals.append((interval_start, len(t) - 1))
    else:
        # No spikes, entire simulation is one interval
        intervals.append((0, len(t) - 1))

    return intervals


def inter_spike_subthreshold(v_m, min_amplitude=0.01):
    peaks, _ = find_peaks(v_m)
    peaks_values = v_m[peaks]

    troughs, _ = find_peaks(-v_m)
    troughs_values = v_m[troughs]

    min_len = min(len(peaks), len(troughs))
    if min_len > 0:
        # Ensure we use the same number of peaks and troughs
        peaks = peaks[:min_len]
        troughs = troughs[:min_len]
        peaks_values = peaks_values[:min_len]
        troughs_values = troughs_values[:min_len]

        amplitudes = np.abs(peaks_values - troughs_values) / 2
        valid_idx = amplitudes >= min_amplitude

        if np.any(valid_idx):
            valid_amplitudes = amplitudes[valid_idx]
            return valid_amplitudes

    return []


def compute_sto(v_m, t, spike_times):
    intervals = get_inter_spike_intervals(v_m, t, spike_times)
    oscillations = []

    for interval in intervals:
        interval_vm = v_m[interval[0] : interval[1]]
        interval_oscillations = inter_spike_subthreshold(interval_vm)
        oscillations.extend(interval_oscillations)

    total_duration_indices = sum(interval[1] - interval[0] for interval in intervals)
    no_spike_time_seconds = total_duration_indices / 10000

    # Handle empty oscillations array
    if len(oscillations) > 0:
        sto_freq = (
            len(oscillations) / no_spike_time_seconds
            if no_spike_time_seconds > 0
            else 0
        )
        sto_amp = np.mean(oscillations)
        sto_std = np.std(oscillations)
    else:
        sto_freq = 0
        sto_amp = 0
        sto_std = 0

    return sto_freq, sto_amp, sto_std


def analyze(vm, sr, milliseconds):
    vm_values = vm.events["V_m"]
    times = vm.events["times"]

    sr_spike_times = sr.events["times"]
    firing_rate = compute_firing_rate(sr_spike_times, milliseconds)
    mean_isi = compute_isi(sr_spike_times)

    sto_freq, sto_amp, sto_std = compute_sto(vm_values, times, sr_spike_times)

    return [
        round(firing_rate, 3),
        round(mean_isi, 3),
        round(sto_amp, 3),
        round(sto_freq, 3),
        round(sto_std, 3),
    ]
