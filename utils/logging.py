"""
Convenience functions for printing information to the console and/or to file
"""

import io
import time
from pathlib import Path
from typing import Optional, Union


class Logger:
    def __init__(self, logfile, print_logs, prefix='', csv_separator='\t'):
        if logfile is not None:
            if type(logfile) == str:
                logfile = Path(logfile).expanduser()
            logfile.parent.mkdir(parents=True, exist_ok=True)

        self.logfile = Path(logfile).expanduser() if logfile is not None else None
        self.print_logs = print_logs

        if len(prefix) > 0 and prefix[-1] != ' ':
            prefix += ' '
        self.prefix = prefix
        self.csv_separator = csv_separator

    def log(self, msg, timestamp=False, prefix=None, file_only=False, print_only=False):
        if type(msg) is not str:
            msg = repr(msg)

        prefix = self.prefix if prefix is None else prefix
        if len(prefix) > 0 and prefix[-1] != ' ':
            prefix += ' '

        if timestamp is not False:
            if timestamp is True or timestamp in ['full']:
                msg = f'[{time.asctime()}] ' + msg
            elif timestamp in ['short', 'clock']:
                msg = f'[{time.asctime()[11:19]}] ' + msg
        msg = prefix + msg
        log(msg, None if print_only else self.logfile, self.print_logs and not file_only)

    def log_csv(self, values, file_only=False, print_only=False, console_col_width=20):
        if file_only and print_only:
            return

        # print human-readable to console
        if not file_only:
            line = ''
            for val in values:
                if type(val) == int:
                    s = f'{val:d}'.rjust(console_col_width)[:console_col_width]
                elif type(val) == str:
                    s = val.ljust(console_col_width)[:console_col_width]
                else:
                    try:
                        s = f'{val:.16f}'.rjust(console_col_width)[:console_col_width]
                    except ValueError:
                        s = str(val).rjust(console_col_width)[:console_col_width]
                line += s + '  '
            self.log(line, print_only=True)

        # print CSV to file
        if not print_only:
            line = ''
            for n, val in enumerate(values):
                line += str(val)
                if n < len(values) - 1:
                    line += self.csv_separator
            self.log(line, file_only=True)

    def warn(self, msg, force_print=False):
        print_logs = force_print or self.print_logs
        log('[WARNING] ' + self.prefix + msg, self.logfile, print_logs)

    def vline(self, file_only=False, print_only=False, length=40):
        self.log('-' * length, prefix=None, file_only=file_only, print_only=print_only)

    def lineskip(self, file_only=False, print_only=False):
        self.log('', prefix=None, file_only=file_only, print_only=print_only)

    def copy(self, prefix: Optional[str] = None):
        prefix = self.prefix if prefix is None else prefix
        return Logger(self.logfile, self.print_logs, prefix)


def log(msg, logfile, print_logs):
    if logfile is not None:
        with io.open(logfile, 'a', buffering=1, newline='\n') as lf:
            lf.write(msg + '\n')
    if print_logs:
        print(msg)


def get_display_fun(logger: Logger) -> callable:
    def disp_fun(*args):
        if len(args) == 0:
            logger.lineskip()

        for msg in args:
            logger.log(msg)

    return disp_fun


def read_logs(logger: Logger, ignore_no_file=False) -> Union[str, None]:
    if logger.logfile is None:
        if ignore_no_file:
            return None
        else:
            raise ValueError('Logger has no logfile')

    if not logger.logfile.exists():
        if ignore_no_file:
            return ''
        else:
            raise ValueError('logfile does not exist at this time')

    # noinspection PyTypeChecker
    with open(logger.logfile, 'rb') as f:
        logs = str(f.read())

    return logs
