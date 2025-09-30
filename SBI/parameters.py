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
    "gsl_error_tol": 1e-5,
}

CONSTANT_PARAMETERS = {
    "refr_t": 50.0,
    "V_reset": -80,
    "E_L": -55,
    "V_m": -55,
    "V_peak": -0,
}

# USED TO GENERATE PRIORS WITH UNIFORM DISTRIBUTION
PARAMETERS_MIN_MAX = {
    "C_m": [0, 500],
    "g_L": [-10, 10],
    "I_e": [-500, +500],
    "Delta_T": [0, 50],
    "a": [0, 8000.0],
    "b": [0, 100.0],
    "tau_w": [0, 8000.0],
    "V_th": [-55, -50],
    "V_peak": [-50, 0],
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
