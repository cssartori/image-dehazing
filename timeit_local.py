#! python3
"""
A timeit replacement that can use the "with" syntax.

When imported, overwrites the built-in timeit().
"""
#pylint: disable= too-few-public-methods

from datetime import datetime
import numpy as np
from typing import Callable, Any
class timeit():
    """
    Helpful timer method. Overwrites the built-in timeit() with this one compatible with the "with" syntax.

    Parameters
    ---------------------
    name: str (default: "Execution Time")
        A string to prepend to the timer. The string is always suffixed with a colon.

    logFn: callable (default= print)
    """

    def __init__(self, name:str= "Execution time", logFn:Callable[[str], Any]= print):
        self.name = name
        self.logFn = logFn
        self.tic = None
    def __enter__(self):
        self.tic = datetime.now()
    def __exit__(self, *args, **kwargs):
        humanTime = int((datetime.now() - self.tic).total_seconds() * 1000)
        formattedTime = str(humanTime) + "ms"
        if humanTime > 1000:
            # Convert to seconds
            humanTime = np.around(humanTime / 1000, 2)
            formattedTime = str(humanTime) + "s"
            if humanTime > 60:
                # Convet to minutes
                minutes = int(humanTime // 60)
                seconds = np.around(humanTime % 60, 2)
                formattedTime = "{0}m {1}s".format(minutes, seconds)
                if minutes > 60:
                    # Convert to hours
                    hours = int(minutes // 60)
                    minutes -= hours * 60
                    formattedTime = "{0}h {1}m {2}s".format(hours, minutes, seconds)
                    if hours > 24:
                        # Convert to days, finally
                        days = int(hours // 24)
                        formattedTime = f"{days}d {hours - days * 24}h {minutes}m {seconds}s"
        self.logFn(f'{self.name}: {formattedTime}')
