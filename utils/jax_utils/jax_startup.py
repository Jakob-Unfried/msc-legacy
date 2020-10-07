"""
configure jax at startup
"""

from jax.config import config


def startup():
    # Jax config (needs to be executed right at startup)
    config.update("jax_enable_x64", True)


def debugging():
    config.update("jax_enable_x64", True)
    config.update("jax_debug_nans", True)
