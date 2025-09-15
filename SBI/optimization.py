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
    n_sims = 10000
    obs_dict = {"firing_rate": 1.0, "vm_mean": -55.0, "STO_fr": 7.0, "STO_amp": 10.0}

    # CREATE PARAMETERS
    priors = create_priors()

    # SAMPLES
    samples = priors.sample((n_sims,))
    indices = list(range(n_sims))
    results = []

    # SIMULATE
    pbar = tqdm(total=n_sims, desc="Initial simulations")

    for i, idx in enumerate(indices):
        parameters = samples[idx]
        result = simulate(parameters)
        results.append(result)

        if i % max(1, len(indices) // 10) == 0:
            pbar.update(max(1, len(indices) // 10))

    pbar.close()

    x = torch.tensor(results, dtype=torch.float32)

    samples = torch.squeeze(samples)
    print(
        f"Completed {n_sims} simulations. Final result shapes: x = {x.shape} || theta = {samples.shape}"
    )

    # RUNNING INFERENCE
    priors, num_parameters, prior_returns_numpy = process_prior(priors)
    simulator = process_simulator(simulate, priors, prior_returns_numpy)
    inference = NPE(prior=priors)

    density_estimator = inference.append_simulations(samples, x).train(
        show_train_summary=True
    )
    posterior = inference.build_posterior(density_estimator)

    print("POSTERIOR: ", posterior)

    # POSTERIOR SAMPLING
    obs = torch.tensor(list(obs_dict.values()), dtype=torch.float32)
    posterior_samples = posterior.sample((n_sims,), x=obs)

    distances = []
    sim_results = []
    mses = []

    for i in indices:
        sim_result = simulate(posterior_samples[i])
        sim_results.append(sim_result)
        error = sim_result - obs.numpy()
        distance = np.linalg.norm(error)
        mse = np.mean(error**2)

        distances.append(distance)
        mses.append(mse)

    distances = torch.tensor(distances, dtype=torch.float32)
    sim_results = torch.tensor(sim_results, dtype=torch.float32)
    mses = torch.tensor(mses, dtype=torch.float32)
    posterior_samples = torch.squeeze(posterior_samples)

    # Trova il theta con distanza minima
    print(
        f"Completed {n_sims} samples. Final result shapes: x_obs = {sim_results.shape} || theta = {posterior_samples.shape}"
    )

    best_distance_idx = torch.argmin(distances)
    best_idx = torch.argmin(mses)
    print(f"Best index: {best_idx} and posterior samples: {posterior_samples.shape}")
    print(
        f"Best distance: {best_distance_idx} and relative theta: {posterior_samples[best_distance_idx]}"
    )
    best_theta = posterior_samples[best_idx]
    best_sim_output = sim_results[best_idx]
    print("Best Theta:", best_theta)
    print("Best Output:", best_sim_output)

    simulate(best_theta, plot=True)
