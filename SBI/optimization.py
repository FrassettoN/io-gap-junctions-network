import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from datetime import datetime
import nest

from sbi.inference import NPE, NLE, NRE
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)

from simulate import simulate
from parameters import create_priors
from analyze_optimization import boxplot, corner_plot, pairplot
from load import load_training_data
from network import simulate_network


def NRE_sequential_inference(prior, obs, simulator, n_sims, rounds):
    inference = NRE(prior)
    proposal = prior
    for _ in range(rounds):
        theta = proposal.sample((n_sims,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        proposal = posterior.set_default_x(obs)

    return posterior


def NLE_sequential_inference(prior, obs, simulator, n_sims, rounds):
    inference = NLE(prior)
    proposal = prior

    for _ in range(rounds):
        theta = proposal.sample((n_sims,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        proposal = posterior.set_default_x(obs)

    return posterior


def NPE_sequential_inference(prior, obs, simulator, n_sims, rounds):
    inference = NPE(prior=prior)
    proposal = prior

    for _ in range(rounds):
        theta = proposal.sample((n_sims,))
        x = simulator(theta)
        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train(show_train_summary=True)
        posterior = inference.build_posterior(density_estimator).set_default_x(obs)
        proposal = posterior

    return posterior


def optimize(
    inference_type,
    n_sims,
    rounds,
    obs_dict,
    obs_weights=None,
):
    # Create simulation name with timestamp and key parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_name = f"n{n_sims}_r{rounds}_{inference_type}_{timestamp}"
    results_dir = os.path.join("results", simulation_name)
    os.makedirs(results_dir, exist_ok=True)

    prior = create_priors()
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    obs = torch.tensor(list(obs_dict.values()), dtype=torch.float32)
    simulator = process_simulator(simulate, prior, prior_returns_numpy)

    if inference_type == "NPE":
        posterior = NPE_sequential_inference(
            prior, obs, simulator, n_sims=n_sims, rounds=rounds
        )
    elif inference_type == "NRE":
        posterior = NRE_sequential_inference(
            prior, obs, simulator, n_sims=n_sims, rounds=rounds
        )
    elif inference_type == "NLE":
        posterior = NLE_sequential_inference(
            prior, obs, simulator, n_sims=n_sims, rounds=rounds
        )
    else:
        print("Inference type not found")
        return 0

    print("POSTERIOR: ", posterior)

    weights = (
        torch.tensor(list(obs_weights.values()), dtype=torch.float32)
        if obs_weights
        else torch.ones_like(obs)
    )

    # POSTERIOR SAMPLING
    posterior_samples = posterior.sample((n_sims,), x=obs)

    distances = []
    mses = []
    weighted_mses = []
    normalized_distances = []

    pbar = tqdm(total=len(posterior_samples), desc="Posterior simulations")

    # First pass: collect all simulation results to calculate min-max statistics
    sim_results = []
    for i in range(len(posterior_samples)):
        sim_result = simulate(posterior_samples[i])
        sim_results.append(sim_result)
        if i % max(1, len(posterior_samples) // 20) == 0:
            pbar.update(max(1, len(posterior_samples) // 20))

    # Calculate min-max normalization statistics
    sim_results = np.array(sim_results)
    sim_min = np.min(sim_results, axis=0)
    sim_max = np.max(sim_results, axis=0)
    sim_range = sim_max - sim_min
    sim_range = np.where(sim_range == 0, 1, sim_range)  # Avoid division by zero

    # Normalize scalar observations to [0, 1] range
    obs_normalized = (obs.numpy() - sim_min) / sim_range
    obs_normalized = np.clip(obs_normalized, 0, 1)  # Ensure within [0, 1]

    pbar.set_description("Processing results with min-max normalization")

    for i in range(len(posterior_samples)):
        sim_result = sim_results[i]

        # Min-max normalized error calculation
        sim_normalized = (sim_result - sim_min) / sim_range
        sim_normalized = np.clip(sim_normalized, 0, 1)  # Ensure within [0, 1]
        normalized_error = sim_normalized - obs_normalized
        normalized_distance = np.linalg.norm(normalized_error)

        # Weighted error calculation (on normalized data)
        weighted_error = normalized_error * weights.numpy()
        weighted_mse = np.mean(weighted_error**2)
        weighted_distance = np.linalg.norm(weighted_error)

        distances.append(weighted_distance)  # Use weighted distance on normalized data
        normalized_distances.append(normalized_distance)
        weighted_mses.append(weighted_mse)

        if i % max(1, len(posterior_samples) // 10) == 0:
            pbar.update(max(1, len(posterior_samples) // 10))

    pbar.close()

    distances = torch.tensor(np.array(distances), dtype=torch.float32)
    sim_results = torch.tensor(sim_results, dtype=torch.float32)
    weighted_mses = torch.tensor(np.array(weighted_mses), dtype=torch.float32)

    # Find best parameters using weighted metrics
    best_weighted_idx = torch.argmin(weighted_mses)
    best_theta = posterior_samples[best_weighted_idx]
    best_sim_output = sim_results[best_weighted_idx]

    print("Best Theta (weighted):", best_theta)
    print("Best Output (weighted):", best_sim_output)

    posterior_data = {
        "posterior_samples": posterior_samples,
        "sim_results": sim_results,
        "distances": distances,
        "weighted_mses": weighted_mses,
        "best_weighted_idx": best_weighted_idx,
        "best_theta": best_theta,
        "best_sim_output": best_sim_output,
    }
    torch.save(posterior_data, os.path.join(results_dir, "posterior_analysis.pt"))

    print("Plotting...")
    pairplot(posterior_samples, best_theta, results_dir)
    boxplot(sim_results, obs_dict, best_sim_output, results_dir)
    corner_plot(posterior_samples, best_theta, results_dir)

    print("Simulating single neuron...")
    simulate(best_theta, results_dir)
    print("Simulating network with gap junctions...")
    simulate_network(best_theta, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Parameter Optimization")
    parser.add_argument(
        "-n",
        "--n_sims",
        type=int,
        default=10000,
        help="Number of simulations to run (default: 10000)",
    )
    parser.add_argument(
        "-i",
        "--inference",
        type=str,
        default="NPE",
        help="Type of inference method to use: NPE, NLE, or NRE (default: NPE)",
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=1,
        help="Number of inference rounds to run (default: 1)",
    )
    args = parser.parse_args()

    n_sims = args.n_sims
    rounds = args.rounds
    inference_type = args.inference
    print(
        f"Running {inference_type} optimization with {n_sims} simulations for {rounds} runs"
    )

    obs_dict = {
        "firing_rate": 1,
        "mean_isi": 1000,
        "STO_amp": 9.5,
        "STO_freq": 4.5,
    }
    obs_weights = {
        "firing_rate": 1.0,
        "mean_isi": 1.0,
        "STO_amp": 1.0,
        "STO_freq": 1.0,
    }

    optimize(
        inference_type,
        n_sims,
        rounds,
        obs_dict,
        obs_weights=obs_weights,
    )
