import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import corner

from parameters import PARAMETERS_MIN_MAX


def boxplot(sim_results, obs_dict, best_sim_output):
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
    plt.savefig(f"posterior_predictive_check_{sim_matrix.shape[0]}.png")


def corner_plot(posterior_samples, best_theta):
    theta_matrix = posterior_samples.numpy()
    print(best_theta)
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
    fig.savefig(f"posterior_corner_plot_{theta_matrix.shape[0]}.png")
