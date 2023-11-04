import csv
import numpy as np
import pandas as pd
from matplotlib import style, pyplot as plot
import seaborn as sb
import sys
import math

class Base_Fitting:
    x = 0.0
    y = 0.0
    delta = sys.float_info.max
    ideal_function = "N/A"

    def __init__(self, x: float, y:float, delta:float, ideal_function: str) -> None:
        self.x = x
        self.y = y
        self.delta = delta
        self.ideal_function = ideal_function

    def __str__(self) -> str:
        """Returns instance data in a human-readable form"""
        return "x: {0}, y: {1}, delta: {2}, ideal function: {3}".format(self.x, self.y, self.delta, self.ideal_function)