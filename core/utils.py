import inspect
from typing import List


def get_config_params(obj) -> List[str]:
    """
    Return configurable parameters in 'Agent.__init__' and 'Model.__init__' which appear after 'config'
    :param obj: An instance of 'Agent' or 'Model'
    :return: A list of configurable parameters
    """
    import core  # Import inside function to avoid cyclic import

    if not isinstance(obj, core.Agent) and not isinstance(obj, core.Model):
        raise ValueError("Only accepts 'Agent.__init__' or 'Model.__init__'")

    sig = list(inspect.signature(obj.__init__).parameters.keys())

    config_params = []
    config_part = False
    for param in sig:
        if param == 'config':
            # Following parameters should be what we want
            config_part = True
        elif param in {'args', 'kwargs'}:
            pass
        elif config_part:
            config_params.append(param)

    return config_params
