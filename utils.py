import pandas as pd
import numpy as np

def readfile(path, filename, separator=";"):
    if filename.endswith(".csv"):
        df = pd.read_csv(path + filename, separator)
    elif filename.endswith(".json"):
        df = pd.read_json(path + filename)
    return df
