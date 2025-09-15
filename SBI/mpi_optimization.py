from mpi4py import MPI
import numpy as np
import torch
from tqdm import tqdm

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from simulate import simulate
from parameters import create_priors


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_sims = 1000
    obs_dict = {"firing_rate": 1.0, "vm_mean": -55.0, "STO_fr": 7.0, "STO_amp": 10.0}

    priors = create_priors()

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

        obs = torch.tensor(list(obs_dict.values()), dtype=torch.float32)

        posterior_samples = posterior.sample((n_sims,), x=obs)
    else:
        posterior_samples = None

    posterior_samples = comm.bcast(posterior_samples, root=0)

    distances = []
    all_sim_results = []
    mses = []

    for i in local_indices:
        sim_result = simulate(posterior_samples[i])
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
        print("Best Theta:", best_theta)
        print("Best Output:", best_sim_output)
