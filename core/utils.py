import inspect
from typing import Callable, List

import core


def get_config_params(init_func: Callable) -> List[str]:
    """
    Return configurable parameters in 'Agent.__init__' and 'Model.__init__' which appear after 'config'
    :param init_func: 'Agent.__init__' or 'Model.__init__'
    :return: A list of configurable parameters
    """

    if init_func is not core.Agent.__init__ and init_func is not core.Model.__init__:
        raise ValueError("Only accepts 'Agent.__init__' or 'Model.__init__'")

    sig = list(inspect.signature(init_func).parameters.keys())

    config_params = []
    config_part = False
    for param in sig:
        if param == 'config':
            # Following parameters should be what we want
            config_part = True
        elif config_part:
            config_params.append(param)

    return config_params

#
# def set_config_params(obj: object, config_params: List[str], config: dict, default_values: dict) -> dict:
#     """
#     Set configurations in following order
#     1. Configurations in 'config'
#     2. Configurations set in 'config_params' argument
#     3. Default configurations
#     :param obj: 'Agent' or 'Model' instance
#     :param config_params: A list of configurable parameters
#     :param config: Configuration file
#     :param default_values: Default values for 'config_params'
#     :return: Configured parameters
#     """
#     param_dict = {p: None for p in config_params}
#
#     for p in param_dict.keys():
#         if config is not None and p in config:
#             param_dict[p] = config[p]
#         else:
#             param_dict[p] = default_values[p]
#
#
#     for p, val in param_dict.items():
#         if val is None:
#             param_dict[p] = default_values[p]
#
#     return param_dict
