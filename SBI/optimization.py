import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from datetime import datetime
import nest

from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)

from simulate import simulate_adex, simulate_eglif
from analyze_optimization import (
    process_posterior_samples_with_normalization,
    boxplot,
    corner_plot,
    pairplot,
)
from save_data import load_training_data, save_optimization_recap
from network import simulate_network
from inference import (
    NRE_sequential_inference,
    NLE_sequential_inference,
    NPE_sequential_inference,
)


def optimize(
    model,
    inference_type,
    n_sims,
    rounds,
    obs_dict,
    obs_weights=None,
):

    if model == "eglif":
        from eglif import (
            create_priors,
            create_parameters_dict,
            PARAMETERS_MIN_MAX,
            CONSTANT_PARAMETERS,
        )
    elif model == "adex":
        from adex import (
            create_priors,
            create_parameters_dict,
            PARAMETERS_MIN_MAX,
            CONSTANT_PARAMETERS,
        )

    # Create simulation name with timestamp and key parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_name = f"{model}_n{n_sims}_r{rounds}_{inference_type}_{timestamp}"
    results_dir = os.path.join("results", simulation_name)
    os.makedirs(results_dir, exist_ok=True)

    prior = create_priors()
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    obs = torch.tensor(list(obs_dict.values()), dtype=torch.float32)

    if model == "eglif":
        simulate = simulate_eglif
    elif model == "adex":
        simulate = simulate_adex

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

    pbar.close()

    distances, sim_results, weighted_mses, best_weighted_idx = (
        process_posterior_samples_with_normalization(
            posterior_samples, sim_results, obs, weights
        )
    )

    best_theta = posterior_samples[best_weighted_idx]
    best_sim_output = sim_results[best_weighted_idx]
    best_theta_dict = create_parameters_dict(best_theta)

    print("Best Theta (weighted):", best_theta_dict)
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
    pairplot(model, posterior_samples, best_theta, results_dir)
    boxplot(model, sim_results, obs_dict, best_sim_output, results_dir)
    corner_plot(model, posterior_samples, best_theta, results_dir)

    print("Simulating single neuron...")
    simulate(best_theta, results_dir)
    print("Simulating network with gap junctions...")
    simulate_network(model, best_theta, results_dir)

    # Save Recap
    save_optimization_recap(
        model,
        results_dir,
        inference_type,
        n_sims,
        rounds,
        timestamp,
        obs_dict,
        obs_weights,
        best_theta_dict,
        best_sim_output,
    )


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
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="adex",
        help="Model to optimize. Options: adex, eglif",
    )
    args = parser.parse_args()

    n_sims = args.n_sims
    rounds = args.rounds
    inference_type = args.inference
    model = args.model
    print(
        f"Running {inference_type} optimization on {model} with {n_sims} simulations for {rounds} runs"
    )

    obs_dict = {
        "firing_rate": 1,
        "mean_isi": 1000,
        "STO_amp": 9.5,
        "STO_freq": 4.5,
        "STO_std": 0.1,
    }
    obs_weights = {
        "firing_rate": 1.0,
        "mean_isi": 1.0,
        "STO_amp": 1.0,
        "STO_freq": 1.0,
        "STO_std": 1.0,
    }

    optimize(
        model,
        inference_type,
        n_sims,
        rounds,
        obs_dict,
        obs_weights=obs_weights,
    )
