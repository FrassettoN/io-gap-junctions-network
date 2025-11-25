import matplotlib.pyplot as plt
import nest
import numpy
import pandas as pd
from analyze_simulation import analyze
from utils import save_simulation_plots
import os

CURRENT_PARAMETERS = {
    "A1": 1810.923,
    "A2": 1358.197,
    "C_m": 189,
    "E_L": -45,
    "I_e": -18.101,
    "V_m": -45,
    "V_reset": -45,
    "V_th": -35,
    "k_1": 0.191,
    "k_2": 0.091,
    "k_adap": 1.928,
    "t_ref": 1,
    "tau_m": 11,
    "E_rev1": 0,
    "E_rev2": -80,
}

OPTIMIZED_PARAMETERS = {
    "E_L": -55,
    "t_ref": 50.0,
    "V_m": -55,
    "V_reset": -55,
    "C_m": 236.6,
    "tau_m": 5.56,
    "I_e": 17.556041717529297,
    "k_adap": 8.381210327148438,
    "k_1": 1.60990309715271,
    "k_2": 0.17936500906944275,
    "V_th": -45.87343215942383,
    "A1": 3716.090576171875,
    "A2": 3535.59423828125,
}


def analyze_model(model, evoked):
    nest.ResetKernel()
    nest.resolution = 0.05
    nest.Install("nestml_gap_eglif_cond_alpha_multisyn_mod_module")
    neuron = nest.Create("eglif_cond_alpha_multisyn_mod_nestml", 1)

    # Set Parameters
    if model == "current":
        neuron.set(CURRENT_PARAMETERS)
    elif model == "optimized":
        neuron.set(OPTIMIZED_PARAMETERS)

    # Create recording devices
    vm = nest.Create("voltmeter", params={"interval": 0.1})
    sr = nest.Create("spike_recorder")
    nest.Connect(vm, neuron)
    nest.Connect(neuron, sr)

    # Simulate cf stimulus if studying evoked activity
    input_times = []
    if evoked:
        start = 700
        duration = 10
        pg = nest.Create(
            "poisson_generator",
            params={"start": start, "stop": start + duration, "rate": 500.0},
        )
        nest.Connect(
            pg, neuron, syn_spec={"weight": 55, "delay": 0.1, "receptor_type": 1}
        )
        input_times.append((start, start + duration))

    # Run simulation
    simulation_time = 2000.0
    nest.Simulate(simulation_time)

    # Analyze and Save results
    results_dir = f"./{'evoked' if evoked else 'spontaneous'}/{model}"
    os.makedirs(results_dir, exist_ok=True)

    results = analyze(vm, sr, simulation_time)
    # Save plots and df if directory provided and results exist
    if results_dir and results:
        save_simulation_plots(vm, sr, results_dir, model, input_times, simulation_time)

        columns = ["firing_rate", "mean_isi", "sto_amp", "sto_freq", "sto_std"]
        # Create DataFrame
        df = pd.DataFrame([results], columns=columns)
        df.to_csv(f"{results_dir}/analysis.csv")


models = ["current", "optimized"]
for model in models:
    analyze_model(model, evoked=False)
    analyze_model(model, evoked=True)
