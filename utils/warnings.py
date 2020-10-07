import sys


def custom_warn(msg, category=UserWarning, filename='', lineno=-1, *args, **kwargs):
    print(f'{category.__name__}: {msg}', file=sys.stderr, flush=True)
    print(f'    issued from: {filename}:{lineno}', file=sys.stderr, flush=True)
