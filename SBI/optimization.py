import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from datetime import datetime
import nest

from sbi.inference import NPE
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)

from simulate import simulate
from parameters import create_priors
from analyze_optimization import boxplot, corner_plot, pairplot
from load import load_training_data


def optimize(n_sims, obs_dict, obs_weights=None, samples=None, x=None):
    # Create simulation name with timestamp and key parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_name = f"n{n_sims}_{timestamp}"

    results_dir = os.path.join("results", simulation_name)
    os.makedirs(results_dir, exist_ok=True)

    # CREATE PARAMETERS
    priors = create_priors()
    indices = list(range(n_sims))

    if samples is None or x is None or samples.shape[0] != n_sims:
        # SAMPLES
        samples = priors.sample((n_sims,))
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

        data_to_save = {
            "samples": samples,
            "x": x,
            "obs_dict": obs_dict,
            "obs_weights": obs_weights,
            "n_sims": n_sims,
            "timestamp": timestamp,
        }

        torch.save(data_to_save, os.path.join(results_dir, "training_data.pt"))

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

    if obs_weights:
        weights = torch.tensor(list(obs_weights.values()), dtype=torch.float32)
    else:
        weights = torch.ones_like(obs)

    distances = []
    sim_results = []
    mses = []
    weighted_mses = []

    for i in indices:
        sim_result = simulate(posterior_samples[i])
        sim_results.append(sim_result)
        error = sim_result - obs.numpy()

        # Standard distance and MSE
        distance = np.linalg.norm(error)
        mse = np.mean(error**2)

        # Weighted error calculation
        weighted_error = error * weights.numpy()
        weighted_mse = np.mean(weighted_error**2)
        weighted_distance = np.linalg.norm(weighted_error)

        distances.append(weighted_distance)  # Use weighted distance
        mses.append(mse)
        weighted_mses.append(weighted_mse)

    distances = torch.tensor(distances, dtype=torch.float32)
    sim_results = torch.tensor(sim_results, dtype=torch.float32)
    mses = torch.tensor(mses, dtype=torch.float32)
    weighted_mses = torch.tensor(weighted_mses, dtype=torch.float32)

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
        "mses": mses,
        "weighted_mses": weighted_mses,
        "best_weighted_idx": best_weighted_idx,
        "best_theta": best_theta,
        "best_sim_output": best_sim_output,
    }
    torch.save(posterior_data, os.path.join(results_dir, "posterior_analysis.pt"))

    pairplot(posterior_samples, best_theta, results_dir)
    boxplot(sim_results, obs_dict, best_sim_output, results_dir)
    corner_plot(posterior_samples, best_theta, results_dir)

    simulate(best_theta, results_dir)


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
        "-l",
        "--load",
        type=str,
        metavar="DIR",
        help="Load pre-existing training data from specified directory. "
        "Directory should contain 'training_data.pt' file with samples and simulation results. "
        "If provided, skips initial simulation phase and uses existing data for inference.",
    )
    args = parser.parse_args()

    n_sims = args.n_sims
    print(f"Running optimization with {n_sims} simulations")

    if args.load:
        samples, x, _ = load_training_data(args.load)
        print(samples.shape[0])
    else:
        samples, x = None, None

    obs_dict = {
        "firing_rate": 1,
        "STO_fr": 5.0,
        "STO_amp": 9.5,
        "STO_growth": 0.2,
    }
    obs_weights = {
        "firing_rate": 0.0,
        "STO_fr": 1.0,
        "STO_amp": 1.0,
        "STO_growth": 2.0,
    }

    optimize(n_sims, obs_dict, obs_weights=obs_weights, samples=samples, x=x)
