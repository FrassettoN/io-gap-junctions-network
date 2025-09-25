import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import corner
from sbi.analysis import pairplot as sbi_pairplot

from parameters import PARAMETERS_MIN_MAX


def boxplot(sim_results, obs_dict, best_sim_output, results_dir=None):
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


def corner_plot(posterior_samples, best_theta, results_dir):
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


def pairplot(posterior_samples, best_theta, results_dir):
    labels = list(PARAMETERS_MIN_MAX.keys())
    limits = list(PARAMETERS_MIN_MAX.values())
    fig, axes = sbi_pairplot(samples=posterior_samples, points=best_theta,limits=limits, labels=labels)

    if results_dir:
        filename = os.path.join(results_dir, f"pairplot.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
