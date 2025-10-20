import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm
from sbi.analysis import pairplot as sbi_pairplot


def process_posterior_samples_with_normalization(
    posterior_samples, sim_results, obs, weights
):
    """
    Process posterior samples with min-max normalization.
    Takes pre-computed simulation results instead of running simulations.
    Returns distances, sim_results, weighted_mses, and best indices.
    """
    distances = []
    normalized_distances = []
    weighted_mses = []

    # Calculate min-max normalization statistics
    sim_results_array = np.array(sim_results)
    sim_min = np.min(sim_results_array, axis=0)
    sim_max = np.max(sim_results_array, axis=0)
    sim_range = sim_max - sim_min
    sim_range = np.where(sim_range == 0, 1, sim_range)  # Avoid division by zero

    # Normalize scalar observations to [0, 1] range
    obs_normalized = (obs.numpy() - sim_min) / sim_range
    obs_normalized = np.clip(obs_normalized, 0, 1)  # Ensure within [0, 1]

    pbar = tqdm(
        total=len(posterior_samples),
        desc="Processing results with min-max normalization",
    )

    for i in range(len(posterior_samples)):
        sim_result = sim_results_array[i]

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
    sim_results_tensor = torch.tensor(sim_results_array, dtype=torch.float32)
    weighted_mses = torch.tensor(np.array(weighted_mses), dtype=torch.float32)

    # Find best parameters using weighted metrics
    best_weighted_idx = torch.argmin(weighted_mses)

    return distances, sim_results_tensor, weighted_mses, best_weighted_idx


def boxplot(model, sim_results, obs_dict, best_sim_output, results_dir=None):
    if model == "eglif":
        from eglif import PARAMETERS_MIN_MAX
    elif model == "adex":
        from adex import PARAMETERS_MIN_MAX

    sim_matrix = np.vstack(sim_results)
    labels = list(obs_dict.keys())
    obs = list(obs_dict.values())

    matplotlib.use("Agg")

    plt.figure(figsize=(10, 6))
    plt.boxplot(sim_results, labels=labels)
    plt.plot(range(1, len(obs) + 1), obs, "ro", label="Target observation")
    plt.plot(
        range(1, len(obs) + 1),
        best_sim_output.numpy(),
        "bo",
        label="Best parameter fit",
    )
    plt.title("Posterior predictive check")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if results_dir:
        filename = os.path.join(results_dir, "boxplot.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()


def corner_plot(model, posterior_samples, best_theta, results_dir):
    if model == "eglif":
        from eglif import PARAMETERS_MIN_MAX
    elif model == "adex":
        from adex import PARAMETERS_MIN_MAX

    theta_matrix = posterior_samples.numpy()
    corner_labels = list(PARAMETERS_MIN_MAX.keys())

    fig = corner.corner(
        theta_matrix,
        labels=corner_labels,
        truths=best_theta.numpy(),
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 10},
    )
    plt.suptitle("Posterior distribution of θ (corner plot)", fontsize=14)
    plt.tight_layout()

    if results_dir:
        filename = os.path.join(results_dir, f"corner_plot.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()


def pairplot(model, posterior_samples, best_theta, results_dir):
    if model == "eglif":
        from eglif import PARAMETERS_MIN_MAX
    elif model == "adex":
        from adex import PARAMETERS_MIN_MAX

    labels = list(PARAMETERS_MIN_MAX.keys())
    limits = list(PARAMETERS_MIN_MAX.values())
    fig, axes = sbi_pairplot(
        samples=posterior_samples, points=best_theta, limits=limits, labels=labels
    )

    if results_dir:
        filename = os.path.join(results_dir, f"pairplot.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
