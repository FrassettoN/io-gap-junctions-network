from sbi import utils as sbi_utils
import torch
from torch.distributions import Distribution
from sbi.utils.torchutils import BoxUniform


DEFAULT_PARAMETERS = {
    "V_m": -70.6,
    "w": 5,  # Must be 5 https://nest-simulator.readthedocs.io/en/stable/model_details/aeif_models_implementation.html
    "C_m": 281.0,
    "t_ref": 0.0,
    "V_reset": -60.0,
    "E_L": -70.6,
    "g_L": 30.0,
    "I_e": 0.0,
    "Delta_T": 2.0,
    "V_th": -50.4,
    "V_peak": 0.0,
    "a": 4.0,
    "b": 80.5,
    "tau_w": 144.0,
    "gsl_error_tol": 1e-5,
}

CONSTANT_PARAMETERS = {
    "refr_T": 50.0,
    "V_reset": -45,
    "E_L": -45,
    "Delta_T": 2,
    "V_m": -45,
    "V_peak": -45,
}

# USED TO GENERATE PRIORS WITH UNIFORM DISTRIBUTION
PARAMETERS_MIN_MAX = {
    "C_m": [0, 500],
    "g_L": [0, 5],
    "I_e": [-5, +30],
    "a": [0, 1000.0],
    "b": [0, 300.0],
    "tau_w": [0, 1000.0],
    "V_th": [-40, -35],
}


def create_parameters_dict(parameters) -> dict:
    parameters_names = list(PARAMETERS_MIN_MAX.keys())

    parameters_dict = {}
    parameters_dict.update(CONSTANT_PARAMETERS)
    for i, parameter in enumerate(parameters):
        parameter_name = parameters_names[i]
        if parameter_name == "E_L":
            parameters_dict["V_m"] = float(parameter)
        parameters_dict[parameter_name] = float(parameter)

    return parameters_dict


class ConstrainedPrior(Distribution):
    def __init__(self, base_prior, constraint_fn):
        self.base_prior = base_prior
        self.constraint_fn = constraint_fn
        super().__init__()

    def log_prob(self, value):
        mask = self.constraint_fn(value)  # per-sample boolean
        base_lp = self.base_prior.log_prob(value)  # per-sample log prob
        # align shapes and set -inf where constraint fails
        if mask.shape != base_lp.shape:
            mask = mask.reshape(base_lp.shape)
        return torch.where(mask, base_lp, torch.full_like(base_lp, float("-inf")))

    def sample(self, sample_shape=torch.Size()):
        if isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)
        total_needed = sample_shape.numel() if sample_shape else 1
        collected = []
        max_tries = 2000
        tries = 0
        while len(collected) < total_needed and tries < max_tries:
            n_try = max(8, (total_needed - len(collected)) * 4)
            candidates = self.base_prior.sample((n_try,))  # (n_try, D)
            valid_mask = self.constraint_fn(candidates)
            valid = candidates[valid_mask]
            if valid.numel() > 0:
                collected.append(valid)
            tries += 1
        if len(collected) == 0:
            raise RuntimeError("No valid samples found under constraint after tries")
        result = torch.cat(collected, dim=0)[:total_needed]
        return result.reshape(sample_shape + (-1,))


def constraint_function(parameters):
    param_names = list(PARAMETERS_MIN_MAX.keys())

    C_m_idx = param_names.index("C_m")
    g_L_idx = param_names.index("g_L")
    a_idx = param_names.index("a")
    tau_w_idx = param_names.index("tau_w")

    C_m = parameters[..., C_m_idx]
    g_L = parameters[..., g_L_idx]
    a = parameters[..., a_idx]
    tau_w = parameters[..., tau_w_idx]

    return a / g_L > (C_m / g_L) / tau_w


def create_priors():
    priors_min, priors_max = zip(*PARAMETERS_MIN_MAX.values())
    base_prior = BoxUniform(
        low=torch.tensor(priors_min, dtype=torch.float32),
        high=torch.tensor(priors_max, dtype=torch.float32),
    )
    return ConstrainedPrior(base_prior, constraint_function)
