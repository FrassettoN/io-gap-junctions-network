from sbi import utils as sbi_utils
import torch


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
}

CONSTANT_PARAMETERS = {
    "t_ref": 50.0,
    "V_reset": -80,
    "E_L": -54.0,
    "V_m": -54.0,
}

PARAMETERS_MIN_MAX = {
    "C_m": [200, 400],
    "g_L": [0, 5.0],
    "I_e": [-30, 30],
    "Delta_T": [4.0, 15.0],
    "V_th": [-50, -30],
    "a": [10.0, 50.0],
    "b": [0, 50],
    "tau_w": [25, 150],
}


def create_parameters_dict(parameters) -> dict:
    parameters_names = list(PARAMETERS_MIN_MAX.keys())

    parameters_dict = {}
    for i, parameter in enumerate(parameters):
        parameter_name = parameters_names[i]
        if parameter_name == "E_L":
            parameters_dict["V_m"] = float(parameter)
        if parameter_name == "V_th":
            parameters_dict["V_peak"] = float(parameter)
        parameters_dict[parameter_name] = float(parameter)

    parameters_dict.update(CONSTANT_PARAMETERS)

    return parameters_dict


def create_priors():
    priors_min, priors_max = zip(*PARAMETERS_MIN_MAX.values())
    priors = sbi_utils.torchutils.BoxUniform(
        low=torch.tensor(priors_min, dtype=torch.float32),
        high=torch.tensor(priors_max, dtype=torch.float32),
    )
    return priors
