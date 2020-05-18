#!python3
import os
from dehazer import dehazeDirectory #pylint: disable= unresolved-import
try:
    assert os.path.exists("../images")
    dehazeDirectory("../images", "../results", verbose= True)
except (FileNotFoundError, AssertionError):
    dehazeDirectory("images", "results", verbose= True)
