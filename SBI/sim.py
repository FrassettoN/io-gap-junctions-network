import numpy as np
import torch
from mpi4py import MPI
from tqdm import tqdm

import nest

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from utils import plot_vm, plot_sr
from analyze_simulation import analyze

DEFAULT_PARAMETERS = {
    "V_m": -70.6,
    # "w": 5, # Must be 5 https://nest-simulator.readthedocs.io/en/stable/model_details/aeif_models_implementation.html
    "C_m": 281.0,
    # "t_ref": 0.0,
    # "V_reset": -60.0,
    "E_L": -70.6,
    # "g_L": 30.0,
    "I_e": 0.0,
    # "Delta_T": 2.0,
    # "V_th": -50.4,
    # "V_peak" : 0.0,
    "a": 4.0,
    "b": 80.5,
    # "tau_w": 144.0,
}


def create_parameters_dict(parameters) -> dict:
    parameters_names = list(DEFAULT_PARAMETERS.keys())

    parameters_dict = {}
    for i, parameter in enumerate(parameters):
        parameter_name = parameters_names[i]
        parameters_dict[parameter_name] = float(parameter)

    return parameters_dict


def simulate(parameters):
    import nest

    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")

    neuron = nest.Create("aeif_cond_alpha_multisynapse", 1)
    parameters_dict = create_parameters_dict(parameters)
    print(parameters_dict)
    neuron.set(parameters_dict)
    vm = nest.Create("voltmeter", params={"interval": 0.1})
    sr = nest.Create("spike_recorder")
    nest.Connect(vm, neuron)
    nest.Connect(neuron, sr)

    milliseconds = 5000.0
    nest.Simulate(milliseconds)

    results = analyze(vm, sr, milliseconds)
    return results


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_sims = 1000
    param_values = np.array(list(DEFAULT_PARAMETERS.values()), dtype=float)
    scale = 0.5
    default_range = 10  # range for zero parameters
    prior_min = np.where(
        param_values != 0, param_values - scale * param_values, -default_range
    )
    prior_max = np.where(
        param_values != 0, param_values + scale * param_values, default_range
    )
    priors = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )

    if rank == 0:
        samples = priors.sample((n_sims,))
    else:
        samples = None

    samples = comm.bcast(samples, root=0)
    local_indices = list(range(rank, n_sims, size))
    local_results = []

    # Add progress bar - only show on rank 0
    if rank == 0:
        pbar = tqdm(total=n_sims, desc="Initial simulations")

    for i, idx in enumerate(local_indices):
        parameters = samples[idx]
        result = simulate(parameters)
        local_results.append(result)

        # Update progress bar periodically
        if rank == 0 and i % max(1, len(local_indices) // 10) == 0:
            pbar.update(size * max(1, len(local_indices) // 10))

    if rank == 0:
        pbar.close()

    local_results = torch.tensor(local_results, dtype=torch.float32)
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        x = torch.cat(all_results, dim=0)
        samples = torch.squeeze(samples)
        print(
            f"Completed {n_sims} simulations. Final result shapes: x = {x.shape} || theta = {samples.shape}"
        )
        priors, num_parameters, prior_returns_numpy = process_prior(priors)
        simulator = process_simulator(simulate, priors, prior_returns_numpy)
        inference = NPE(prior=priors)

        density_estimator = inference.append_simulations(samples, x).train(
            show_train_summary=True
        )
        posterior = inference.build_posterior(density_estimator)

        print("POSTERIOR: ", posterior)

        obs = torch.tensor([1, -65, 7, 10], dtype=torch.float32)

        posterior_samples = posterior.sample((1,), x=obs)
        print(posterior_samples)
    else:
        posterior_samples = None

    posterior_samples = comm.bcast(posterior_samples, root=0)

    distances = []
    all_sim_results = []
    mses = []

    for i in local_indices:
        sim_result = simulate(posterior_samples[i])
        print(sim_results)
        all_sim_results.append(sim_result)
        error = sim_result - obs.numpy()
        distance = np.linalg.norm(error)
        mse = np.mean(error**2)

        distances.append(distance)
        mses.append(mse)

    distances = torch.tensor(distances, dtype=torch.float32)
    all_distances = comm.gather(distances, root=0)
    all_sim_results = torch.tensor(all_sim_results, dtype=torch.float32)
    total_sims = comm.gather(all_sim_results, root=0)
    mses = torch.tensor(mses, dtype=torch.float32)
    all_mses = comm.gather(mses, root=0)

    if rank == 0:
        distances = torch.cat(all_distances, dim=0)
        posterior_samples = torch.squeeze(posterior_samples)
        total = torch.cat(total_sims, dim=0)
        mses = torch.cat(all_mses, dim=0)
        # Trova il theta con distanza minima
        print(
            f"Completed {n_sims} samples. Final result shapes: x_obs = {total.shape} || theta = {posterior_samples.shape}"
        )

        best_distance_idx = torch.argmin(distances)
        best_idx = torch.argmin(mses)
        print(
            f"Best index: {best_idx} and posterior samples: {posterior_samples.shape}"
        )
        print(
            f"Best distance: {best_distance_idx} and relative theta: {posterior_samples[best_distance_idx]}"
        )
        best_theta = posterior_samples[best_idx]
        best_sim_output = total[best_idx]
        print("Miglior theta trovato:", best_theta)
        print("Output simulato con miglior theta:", best_sim_output)
