from scipy.signal import find_peaks
import numpy as np
import scipy.signal as signal


def compute_firing_rate(spike_times, milliseconds):
    seconds = milliseconds / 1000.0

    return len(spike_times) / seconds


def compute_isi(spike_times):
    """
    Compute Inter-Spike Interval statistics.

    Parameters:
    -----------
    spike_times : array-like
        Array of spike times in milliseconds

    Returns:
    --------
        - mean_isi: Mean inter-spike interval in ms

    """
    if len(spike_times) < 2:
        return 0.0

    # Calculate intervals between consecutive spikes
    intervals = np.diff(spike_times)

    mean_isi = np.mean(intervals)

    return mean_isi


def compute_sto(v_m, t, milliseconds, V_th, spike_times=None, min_amplitude=0.01):
    peaks, _ = find_peaks(v_m)
    troughs, _ = find_peaks(-v_m)

    if spike_times is not None and len(spike_times) > 0:
        # Convert spike times to indices
        spike_indices = np.searchsorted(t, spike_times)

        # Concatenate peaks and troughs with their types
        all_extrema = np.concatenate([peaks, troughs])
        extrema_types = np.concatenate(
            [np.ones(len(peaks)), np.zeros(len(troughs))]
        )  # 1=peak, 0=trough

        # Sort by time index
        sort_idx = np.argsort(all_extrema)
        all_extrema = all_extrema[sort_idx]
        extrema_types = extrema_types[sort_idx]

        # Remove spike peaks and their following troughs
        keep_mask = np.ones(len(all_extrema), dtype=bool)

        for spike_idx in spike_indices:
            # Find the spike peak (closest peak to spike time)
            peak_distances = np.abs(all_extrema - spike_idx)
            if not peak_distances.any():
                return 0, 0, 0
            spike_peak_pos = np.argmin(peak_distances)

            # Only consider if it's actually a peak
            if extrema_types[spike_peak_pos] == 1:
                keep_mask[spike_peak_pos] = False  # Remove spike peak

                # Find and remove the next trough after this peak
                for i in range(spike_peak_pos + 1, len(all_extrema)):
                    if extrema_types[i] == 0:  # Found next trough
                        keep_mask[i] = False
                        break

        # Filter back to separate peaks and troughs
        filtered_extrema = all_extrema[keep_mask]
        filtered_types = extrema_types[keep_mask]

        peaks = filtered_extrema[filtered_types == 1]
        troughs = filtered_extrema[filtered_types == 0]

    peaks_values = v_m[peaks]
    troughs_values = v_m[troughs]

    min_len = min(len(peaks), len(troughs))
    if min_len > 0:
        # Ensure we use the same number of peaks and troughs
        peaks = peaks[:min_len]
        troughs = troughs[:min_len]
        peaks_values = peaks_values[:min_len]
        troughs_values = troughs_values[:min_len]

        amplitudes = np.abs(peaks_values - troughs_values)
        valid_idx = amplitudes >= min_amplitude

        if np.any(valid_idx):
            valid_amplitudes = amplitudes[valid_idx]
            valid_peaks = peaks[valid_idx]  # Now these have matching lengths

            valid_milliseconds = milliseconds - (
                50 * len(spike_times) if spike_times is not None else 0
            )
            valid_seconds = valid_milliseconds / 1000.0
            if valid_seconds == 0:
                return 0, 0, 0

            sto_freq = len(valid_amplitudes) / valid_seconds

            mean_amp = np.mean(valid_amplitudes)

            # Calculate amplitude growth between intervals
            sto_growth = 0.0
            if len(valid_amplitudes) > 1:
                growth_values = []

                # Define intervals
                intervals = []
                if spike_times is not None and len(spike_times) > 0:
                    # Start to first spike
                    intervals.append((0, spike_indices[0]))

                    # Between consecutive spikes
                    for i in range(len(spike_indices) - 1):
                        intervals.append((spike_indices[i], spike_indices[i + 1]))

                    # Last spike to end
                    intervals.append((spike_indices[-1], len(t) - 1))
                else:
                    # No spikes, consider entire simulation
                    intervals.append((0, len(t) - 1))

                # Calculate growth for each interval with at least 2 oscillations
                for start_idx, end_idx in intervals:
                    interval_mask = (valid_peaks >= start_idx) & (
                        valid_peaks <= end_idx
                    )
                    interval_amps = valid_amplitudes[interval_mask]

                    if len(interval_amps) > 1:
                        growth = interval_amps[-1] - interval_amps[0]
                        growth_values.append(growth)

                if growth_values:
                    sto_growth = np.mean(growth_values)

            return (sto_freq, mean_amp, sto_growth)

    return 0, 0, 0


def analyze(vm, sr, milliseconds, V_th):
    vm_values = vm.events["V_m"]
    times = vm.events["times"]

    sr_spike_times = sr.events["times"]
    firing_rate = compute_firing_rate(sr_spike_times, milliseconds)
    mean_isi = compute_isi(sr_spike_times)

    sto_freq, sto_amp, sto_growth = compute_sto(
        vm_values, times, milliseconds, V_th, sr_spike_times
    )

    return [
        round(firing_rate, 3),
        round(mean_isi, 3),
        # round(sto_freq, 3),
        # round(sto_amp, 3),
        # round(sto_growth, 3),
    ]
