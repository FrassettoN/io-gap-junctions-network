import os
import torch
import json


def save_optimization_recap(
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
    """
    Save optimization configuration and results to JSON and text files.
    """
    # Convert tensor to list for JSON serialization
    best_sim_output_list = [x.item() for x in best_sim_output]

    recap_data = {
        "simulation_config": {
            "inference_type": inference_type,
            "n_sims": n_sims,
            "rounds": rounds,
            "timestamp": timestamp,
        },
        "observed_data": obs_dict,
        "observation_weights": obs_weights,
        "parameter_ranges": dict(PARAMETERS_MIN_MAX),
        "constant_parameters": dict(CONSTANT_PARAMETERS),
        "best_parameters": best_theta_dict,
        "best_simulation_output": dict(zip(obs_dict.keys(), best_sim_output_list)),
    }

    # Save JSON file
    json_file = os.path.join(results_dir, "optimization_recap.json")
    with open(json_file, "w") as f:
        json.dump(recap_data, f, indent=2)

    # Also save human-readable text file
    txt_file = os.path.join(results_dir, "optimization_recap.txt")
    with open(txt_file, "w") as f:
        f.write("OPTIMIZATION RECAP\n")
        f.write("=" * 50 + "\n\n")

        # Simulation parameters
        f.write(f"Inference Type: {inference_type}\n")
        f.write(f"Number of Simulations: {n_sims}\n")
        f.write(f"Number of Rounds: {rounds}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        # Observed data
        f.write("OBSERVED DATA:\n")
        f.write("-" * 20 + "\n")
        for key, value in obs_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Observation weights
        f.write("OBSERVATION WEIGHTS:\n")
        f.write("-" * 20 + "\n")
        for key, value in obs_weights.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Parameter ranges
        f.write("PARAMETER RANGES:\n")
        f.write("-" * 20 + "\n")
        for key, (min_val, max_val) in PARAMETERS_MIN_MAX.items():
            f.write(f"{key}: [{min_val}, {max_val}]\n")
        f.write("\n")

        f.write("CONSTANT PARAMETERS:\n")
        f.write("-" * 20 + "\n")
        for key, value in CONSTANT_PARAMETERS.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Best parameters
        f.write("BEST PARAMETERS (OPTIMIZED):\n")
        f.write("-" * 30 + "\n")
        for key, value in best_theta_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Best simulation output
        f.write("BEST SIMULATION OUTPUT:\n")
        f.write("-" * 25 + "\n")
        obs_keys = list(obs_dict.keys())
        for i, key in enumerate(obs_keys):
            f.write(
                f"{key}: {best_sim_output[i].item():.4f} (target: {obs_dict[key]})\n"
            )

    print(f"Optimization recap saved to: {json_file} and {txt_file}")


def load_training_data(results_dir):
    """Load saved training data and priors"""
    # Load training data
    data = torch.load(os.path.join(results_dir, "training_data.pt"))
    samples = data["samples"]
    x = data["x"]
    obs_dict = data["obs_dict"]
    obs_weights = data["obs_weights"]

    return samples, x, obs_dict


def load_posterior_analysis(results_dir):
    """Load saved posterior analysis data"""
    data = torch.load(os.path.join(results_dir, "posterior_analysis.pt"))

    best_theta = data["best_theta"]
    best_sim_output = data["best_sim_output"]

    return best_theta, best_sim_output
