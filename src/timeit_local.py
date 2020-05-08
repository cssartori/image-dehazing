#! python3
"""
A timeit replacement that can use the "with" syntax.

When imported, ovewrites the built-in timeit().
"""
#pylint: disable= too-few-public-methods
class timeit():
    """
    Helpful timer method. Overwrites the built-in timeit() with this one compatible with the "with" syntax.

    Parameters
    ---------------------
    name: str (default: "Execution Time")
        A string to prepend to the timer. The string is always suffixed with a colon.
    """
    from datetime import datetime
    import numpy as np
    def __init__(self, name= "Execution time", logFn= print):
        self.name = name
        self.logFn = logFn
    def __enter__(self):
        self.tic = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        import numpy as np
        self.x = self.name
        humanTime = int((self.datetime.now() - self.tic).total_seconds() * 1000)
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
                        formattedTime = "{0}d {1}h {2}m {3}s".format(days, hours - days * 24, minutes, seconds)
        self.logFn('{0}: {1}'.format(self.name, formattedTime))
