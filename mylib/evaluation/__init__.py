import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def print_errors(test, predicted):
    print("MSE: ", mean_squared_error(test, predicted))
    print("RMSE: ", np.sqrt(((test - predicted) ** 2).mean()))
    print("R2: ", r2_score(test, predicted))
    print("RMSE % of mean:", np.sqrt(((test - predicted) ** 2).mean()) / test.mean())
    print("Calibration:", predicted.mean() / test.mean())
