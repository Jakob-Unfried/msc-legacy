from typing import Optional


def parse_options(dict_in: Optional[dict], defaults: Optional[dict] = None):
    """
    Utility function to be used for e.g. kwargs
    1) creates a copy of dict_in, such that it is safe to change its entries
    2) converts None to an empty dictionary (this is useful, since empty dictionaries cant be argument defaults)
    3) optionally, sets defaults, if keys are not present

    Parameters
    ----------
    dict_in
    defaults

    Returns
    -------

    """
    if dict_in is None:
        dict_in = {}
    else:
        dict_in = dict_in.copy()
    if defaults:
        for key in defaults:
            dict_in.setdefault(key, defaults[key])
    return dict_in
