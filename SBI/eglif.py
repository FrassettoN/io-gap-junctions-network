from sbi import utils as sbi_utils
import torch

CONSTANT_PARAMETERS = {
    "E_L": -45,
    "t_ref": 50.0,
    "V_m": -45,
    "V_reset": -45,
    "V_th": -30,
}

# USED TO GENERATE PRIORS WITH UNIFORM DISTRIBUTION
PARAMETERS_MIN_MAX = {
    "C_m": [0, 500],
    "tau_m": [0, 20],
    "I_e": [-20, +20],
    "k_adap": [0, 5],
    "k_1": [0, 1],
    "k_2": [0, 1],
    "V_th": [-40, -30],
    "tau_V": [0, 2],
    "lambda_0": [0, 2],
    "A1": [0, 2500],
    "A2": [0, 2500],
}


def create_parameters_dict(parameters) -> dict:
    parameters_names = list(PARAMETERS_MIN_MAX.keys())

    parameters_dict = {}
    parameters_dict.update(CONSTANT_PARAMETERS)
    for i, parameter in enumerate(parameters):
        parameter_name = parameters_names[i]
        if parameter_name == "E_L":
            parameters_dict["V_m"] = float(parameter)
        # if parameter_name == "V_th":
        #     parameters_dict["V_peak"] = float(parameter)
        parameters_dict[parameter_name] = float(parameter)

    return parameters_dict


def create_priors():
    priors_min, priors_max = zip(*PARAMETERS_MIN_MAX.values())
    priors = sbi_utils.torchutils.BoxUniform(
        low=torch.tensor(priors_min, dtype=torch.float32),
        high=torch.tensor(priors_max, dtype=torch.float32),
    )
    return priors
