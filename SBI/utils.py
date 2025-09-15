import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, find_peaks


def plot_vm(vm, spikes=[], peaks=[], show=False, save=False, output="V_m.png"):
    vm_values = vm.events["V_m"]
    senders = vm.events["senders"]
    times = vm.events["times"]
    plt.figure(figsize=(10, 5))
    for cell_num in np.unique(vm.events["senders"]):
        plt.plot(
            times[np.where(senders == cell_num)],
            vm_values[np.where(senders == cell_num)],
            label=f"Neuron {cell_num}",
        )
    plt.ticklabel_format(style="plain", axis="y", useOffset=False)
    plt.legend(loc="upper right")
    for spike in spikes:
        plt.axvline(
            x=spike,
            linestyle=":",
            linewidth=1.5,
            color="red",
        )
    for peak in peaks:
        plt.axvline(
            x=peak,
            linestyle=":",
            linewidth=1.5,
            color="green",
        )
    plt.xlabel("time (ms)")
    plt.ylabel("membrane potential (mV)")
    if show:
        plt.show()
    if save:
        plt.savefig(output)


def plot_sr(sr, show=False, save=False, output="images/spikes"):
    evs = sr.events["senders"]
    ts = sr.events["times"]
    plt.figure(figsize=(10, 5))
    plt.plot(ts, evs, ".")
    if show:
        plt.show()
    if save:
        plt.savefig(output)
