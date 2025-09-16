import torch


def load_training_data(results_dir):
    """Load saved training data and priors"""
    import pickle

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
