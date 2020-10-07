"""
Fills a kwargs dictionary with default values, optionally displays the kwargs nicely,
then makes all kwargs available as a namespace
"""

from argparse import Namespace

from utils.logging import Logger, get_display_fun


class KwargsParser(Namespace):
    def __init__(self, kwargs: dict, defaults: dict):
        kwargs = kwargs.copy()
        for key in defaults:
            kwargs.setdefault(key, defaults[key])

        super().__init__(**kwargs)

    def log(self, displayer):
        if type(displayer) == Logger:
            displayer = get_display_fun(displayer)

        display_kwargs(self.kwargs(), displayer)

    def kwargs(self):
        return self.__dict__


def display_kwargs(kwargs: dict, display_fun: callable):
    display_fun('kwargs')
    for key in kwargs:
        key_str = str(key)
        padding = max(0, 20 - len(key_str))
        display_fun(f'    {key_str}:{" " * padding}{kwargs[key]}')
