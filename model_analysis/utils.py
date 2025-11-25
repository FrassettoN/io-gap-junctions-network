import matplotlib.pyplot as plt
import numpy as np
import os


import matplotlib.pyplot as plt
import numpy as np


def plot_vm(
    vm,
    sim_duration=2500,
    model="current",
    input_times=None,
    show=False,
    save=False,
    output="V_m.png",
):
    """
    Plot membrane potential (V_m) trace for a single neuron.
    Optionally indicate external input periods (e.g., Poisson generator).
    """
    vm_values = vm.events["V_m"]
    times = vm.events["times"]

    plt.figure(figsize=(10, 4))
    plt.plot(times, vm_values, color="black", linewidth=1.5)

    # Mark input periods if provided
    activity_type = "evoked" if input_times else "spontaneous"
    if input_times:
        for t_start, t_end in input_times:
            plt.axvspan(
                t_start,
                t_end,
                color="red",
                alpha=0.2,
                label="Poisson input (500 Hz, w=55)",
            )
    plt.xlabel("Time (ms)", fontsize=14)
    plt.ylabel("Membrane potential (mV)", fontsize=14)
    plt.title(
        f"{model.title()} - Membrane Potential ({activity_type.title()})",
        fontsize=16,
        weight="bold",
        pad=10,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xlim(0, sim_duration)

    # Show legend only if input is marked
    if input_times:
        plt.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)

    plt.tight_layout()

    if save:
        plt.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_sr(
    sr,
    sim_duration=2500,
    model="current",
    input_times=None,
    show=False,
    save=False,
    output="spikes.png",
):
    """
    Plot spike raster for a single neuron or population.
    Optionally indicate external input periods (e.g., Poisson generator).
    """
    evs = sr.events["senders"]
    ts = sr.events["times"]

    plt.figure(figsize=(10, 1))

    # Single neuron: event plot
    plt.eventplot(ts, colors="black", lineoffsets=1, linelengths=0.6, linewidths=1)
    plt.ylim(0.5, 1.5)
    plt.yticks([])

    # Mark input periods if provided
    activity_type = "evoked" if input_times else "spontaneous"
    if input_times:
        for t_start, t_end in input_times:
            plt.axvspan(
                t_start,
                t_end,
                color="red",
                alpha=0.2,
                label="Poisson input (500 Hz, w=55)",
            )

    # Thesis-quality styling
    plt.xlim(0, sim_duration)
    plt.xlabel("Time (ms)", fontsize=14)
    plt.title(
        f"{model.title()} - Spiking Activity ({activity_type.title()})",
        fontsize=12,
        weight="bold",
        pad=10,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Show legend only if input marked
    if input_times:
        plt.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)

    plt.tight_layout()

    if save:
        plt.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def save_simulation_plots(vm, sr, results_dir, model, input_times, sim_duration):
    """Save voltage and spike plots to results directory."""
    try:
        results_dir = "./figures"
        activity_type = "evoked" if input_times else "spontaneous"
        vm_filename = os.path.join(results_dir, f"{model}_{activity_type}_vt.pdf")
        sr_filename = os.path.join(results_dir, f"{model}_{activity_type}_rp.pdf")

        plot_vm(
            vm,
            sim_duration=sim_duration,
            model=model,
            input_times=input_times,
            save=True,
            output=vm_filename,
        )
        plot_sr(
            sr,
            sim_duration=sim_duration,
            model=model,
            input_times=input_times,
            save=True,
            output=sr_filename,
        )
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")
