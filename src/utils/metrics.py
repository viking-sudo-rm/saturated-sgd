from typing import Union, Dict


def update_metrics(return_dict: Dict[str, float], name: str, value: Union[float, dict]) -> None:
    """Unpack nested metric return values into a flat dictionary."""
    if not isinstance(value, dict):
        return_dict[name] = value
    else:
        for subname, subvalue in value.items():
            update_metrics(return_dict, f"{name}/{subname}", subvalue)
