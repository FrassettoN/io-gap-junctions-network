from sbi import utils as sbi_utils
import torch

CONSTANT_PARAMETERS = {
    "E_L": -55,
    "t_ref": 50.0,
    "V_m": -55,
    "V_reset": -55,
    "C_m": 236.6,
    "tau_m": 5.56,
}

# USED TO GENERATE PRIORS WITH UNIFORM DISTRIBUTION
PARAMETERS_MIN_MAX = {
    "I_e": [-20, +20],
    "k_adap": [5, 10],
    "k_1": [0, 2],
    "k_2": [0, 0.5],
    "V_th": [-50, -40],
    "A1": [0, 5000],
    "A2": [0, 5000],
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


def create_priors():
    priors_min, priors_max = zip(*PARAMETERS_MIN_MAX.values())
    priors = sbi_utils.torchutils.BoxUniform(
        low=torch.tensor(priors_min, dtype=torch.float16),
        high=torch.tensor(priors_max, dtype=torch.float16),
    )
    return priors
